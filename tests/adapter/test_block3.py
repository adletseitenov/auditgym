"""Block 3 (tests 1-20) and the open examples M01-M12 of the published checks, through judge_adapter.

tests/judge/test_official_block3.py is the unit layer (auditgym.judge on hand-built episodes). Here every
case is an episode DIRECTORY on disk, scored by `judge_adapter.score_episode(episode_dir, findings,
inspection_steps)`, and, where the budget allows, also played in closed mode through `make_env()`
(`reset` on the same directory, `inspection_steps` exploratory steps, then `submit`): the evaluator journal
must carry the same TP/FP/FN, the same reward and the same inspection_steps, and the policy must see 0.

Fixtures. The open data is always a real episode made by `judge_adapter.generate_episode`. The page defines
the truth (one error A: type T, R1 (and R2), 10000 KZT, evidence {D1, D2}, D3 an existing extra id, U absent,
tolerance 0), and a generated episode cannot have exactly that truth, so for those cases the truth (and, for
M08, the tolerance of the policy) is rewritten in a COPY of the directory; R1.. are real invoice ids, D1.. are
real ledger rows, U is an id that is absent from every table, and `provenance`/`pre_*` are dropped (the
evaluator never reads them). A truth id that is not in the open data is rejected by the fixture itself.
Tests 3, 4 and the real-oracle cases in RealEpisodeTest use UNEDITED generated episodes (`verify` passes);
there the truth is the real injection.

Expected numbers are written as the page gives them (fractions where it gives fractions); abs tolerance 1e-9.
Pairs of M10/M11 (which finding takes which error) are visible only to the unit layer (`matcher.match`);
through the adapter the invariants are TP/FP/FN and the reward, for both orders of the findings.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import unittest
from fractions import Fraction

import judge_adapter
from auditgym.matcher import known_ids
from auditgym.storage import load_episode
from auditgym.verify import verify
from tests.adapter.helpers import oracle_findings

T = "duplicate_invoice"
TYPES = (T, "allocation_gap", "fx_error", "period_cutoff")
TOL = 1e-9
BASE_SEED = 92001
BASE_CONFIG = {"errors": {T: 1}}
CLEAN_SEED, CLEAN_CONFIG = 92002, {"errors": {}}
TWO_SEED, TWO_CONFIG = 92003, {"errors": {T: 2}}
STRUCTURE_FIELDS = ("tp", "fp", "fn", "precision", "recall", "f1")

_state: dict = {}


def _scratch() -> pathlib.Path:
    if "scratch" not in _state:
        _state["scratch"] = tempfile.TemporaryDirectory()
    return pathlib.Path(_state["scratch"].name)


def tearDownModule():
    for env, _ in _state.get("envs", {}).values():
        env.close()
    if "scratch" in _state:
        _state["scratch"].cleanup()
    _state.clear()


def _generated(name: str, seed: int, config: dict) -> pathlib.Path:
    directory = _scratch() / name
    if not directory.exists():
        judge_adapter.generate_episode(seed, config, str(directory))
    return directory


class Fixture:
    """An episode directory with the page's ids: R (invoices), D (ledger rows), U (absent), episode_id."""

    def __init__(self, directory: pathlib.Path) -> None:
        self.dir = str(directory)
        public = load_episode(directory)["public"]
        self.episode_id = public["manifest"]["episode_id"]
        self.R = [doc["doc_id"] for doc in public["invoices"]][:6]
        self.D = [row["row_id"] for row in public["ledger"]][:12]
        self.known = known_ids(public)
        self.U = next(f"U{n}x" for n in range(10 ** 6) if f"U{n}x" not in self.known)


def _synthetic(name: str, issues: list, tolerance: int = 0) -> Fixture:
    """A copy of the base episode with the truth of the page (and a uniform tolerance)."""
    if name in _state.setdefault("fixtures", {}):
        return _state["fixtures"][name]
    base = _generated("base", BASE_SEED, BASE_CONFIG)
    directory = _scratch() / name
    shutil.copytree(base, directory)
    for leftover in ("private/provenance.json", "private/pre_ledger.csv", "private/pre_invoices.jsonl",
                     "private/pre_payments.jsonl"):
        (directory / leftover).unlink()
    fixture = Fixture(directory)
    truth = {"schema_version": "1", "episode_id": fixture.episode_id, "issues": issues}
    for issue in issues:  # the truth only names real records of the open data
        for identifier in issue["affected_ids"] + [i for group in issue["evidence_sets"] for i in group]:
            assert identifier in fixture.known, identifier
    (directory / "private" / "truth.json").write_text(json.dumps(truth, indent=1) + "\n", encoding="utf-8")
    policy_path = directory / "public" / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["amount_tolerance_minor"] = {kind: tolerance for kind in TYPES}
    policy_path.write_text(json.dumps(policy, indent=1) + "\n", encoding="utf-8")
    _state["fixtures"][name] = fixture
    return fixture


def issue(issue_id, affected, evidence_sets, amount=10000, currency="KZT", kind=T) -> dict:
    return {"issue_id": issue_id, "type": kind, "affected_ids": list(affected),
            "expected_amount_minor": amount, "currency": currency,
            "evidence_sets": [list(group) for group in evidence_sets]}


def single_a() -> Fixture:
    """Block 3: one error A on R1, evidence {D1, D2}, 10000 KZT, tau 0."""
    base = _generated("base", BASE_SEED, BASE_CONFIG)
    first = Fixture(base)
    return _synthetic("single_a", [issue("A", first.R[:1], [first.D[:2]])])


def two_errors() -> Fixture:
    first = Fixture(_generated("base", BASE_SEED, BASE_CONFIG))
    return _synthetic("two_errors", [issue("A", first.R[:1], [first.D[:2]]), issue("B", first.R[1:2], [first.D[:2]])])


def pair_a(tolerance: int = 0, name: str = "pair_a") -> Fixture:
    """M01-M09, M12: one error A with affected_ids {R1, R2}."""
    first = Fixture(_generated("base", BASE_SEED, BASE_CONFIG))
    return _synthetic(name, [issue("A", first.R[:2], [first.D[:2]])], tolerance)


def pair_ab(sets_a, sets_b, name: str) -> Fixture:
    """M10, M11: A and B with the same type, ids, amount and currency; they differ only in evidence sets."""
    first = Fixture(_generated("base", BASE_SEED, BASE_CONFIG))
    sets = {"d12": [first.D[:2]], "d13": [[first.D[0], first.D[2]]]}
    return _synthetic(name, [issue("A", first.R[:2], sets[sets_a]), issue("B", first.R[:2], sets[sets_b])])


def finding(fixture: Fixture, affected=None, evidence=None, amount=10000, currency="KZT", kind=T,
            explanation="ok") -> dict:
    return {"type": kind, "affected_ids": list(fixture.R[:1] if affected is None else affected),
            "claimed_amount_minor": amount, "currency": currency,
            "evidence_ids": list(fixture.D[:2] if evidence is None else evidence), "explanation": explanation}


def formula(tp: int, fp: int, fn: int, steps: int) -> float:
    f1 = Fraction(1) if tp + fp + fn == 0 else Fraction(2 * tp, 2 * tp + fp + fn)
    return float(max(Fraction(-1), min(Fraction(1), 2 * f1 - 1 - Fraction(2, 1000) * steps)))


class AdapterBlock(unittest.TestCase):
    """Helpers: score through score_episode, then play the same answer through make_env."""

    @classmethod
    def log_path(cls) -> pathlib.Path:
        return _scratch() / "score_errors.jsonl"

    def submit(self, fixture: Fixture, findings: list) -> dict:
        return {"episode_id": fixture.episode_id, "findings": findings}

    def env_for(self, fixture: Fixture):
        envs = _state.setdefault("envs", {})
        if fixture.dir not in envs:
            journal = _scratch() / f"journal-{len(envs)}.jsonl"
            envs[fixture.dir] = (judge_adapter.make_env(journal_path=journal), journal)
        return envs[fixture.dir]

    def play(self, fixture: Fixture, payload, steps: int):
        """closed-mode run: reset, `steps` exploratory calculator steps, submit. Returns (observation, flags, row)."""
        env, journal = self.env_for(fixture)
        env.reset(0, {"mode": "closed", "episode_dir": fixture.dir})
        for _ in range(steps):
            observation, reward, terminated, truncated, info = env.step(
                {"name": "calculator", "args": {"expression": "1+1"}})
            self.assertEqual((observation["error"], reward, terminated, truncated, info), (None, 0.0, False, False, {}))
        observation, reward, terminated, truncated, info = env.step({"name": "submit", "args": {"findings": payload}})
        self.assertEqual((reward, terminated, truncated, info), (0.0, True, False, {}))  # the policy sees nothing
        self.assertEqual(env.inspection_steps, steps)
        row = env.last_result()
        self.assertIsNotNone(row)
        self.assertEqual(row["inspection_steps"], steps)
        return observation, row

    def check(self, fixture: Fixture, findings: list, tp: int, fp: int, fn: int, reward, steps: int = 10,
              precision=None, recall=None, f1=None, env: bool = True) -> dict:
        payload = self.submit(fixture, findings)
        result = judge_adapter.score_episode(fixture.dir, payload, steps, journal_path=self.log_path())
        self.assertEqual(sorted(result), sorted(("tp", "fp", "fn", "precision", "recall", "f1", "return")))
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (tp, fp, fn))
        for name, want in (("precision", precision), ("recall", recall), ("f1", f1)):
            if want is not None:
                self.assertAlmostEqual(result[name], float(want), delta=TOL, msg=name)
        self.assertAlmostEqual(result["return"], float(reward), delta=TOL)
        self.assertAlmostEqual(result["return"], formula(tp, fp, fn, steps), delta=TOL)  # the case-wide formula
        if env and steps < 100:  # 100 exploratory steps leave no action for the submit: formula test only
            observation, row = self.play(fixture, payload, steps)
            self.assertEqual(observation["result"], {"accepted": True})
            self.assertEqual((row["tp"], row["fp"], row["fn"], row["structure_error"]), (tp, fp, fn, None))
            self.assertAlmostEqual(row["reward"], float(reward), delta=TOL)
        return result

    def structure(self, fixture: Fixture, payload, code: str, steps: int = 10) -> None:
        """An invalid structure: return -1, the other fields null, the code in the evaluator's score log, and the
        closed run ends (terminated) with reward -1 in the evaluator's journal only."""
        log = self.log_path()
        before = log.read_text(encoding="utf-8").count("\n") if log.exists() else 0
        result = judge_adapter.score_episode(fixture.dir, payload, steps, journal_path=log)
        self.assertEqual(result["return"], -1)
        for name in STRUCTURE_FIELDS:
            self.assertIsNone(result[name], name)
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()][before:]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], {"event": "score_error", "episode_id": fixture.episode_id, "structure_error": code})
        observation, row = self.play(fixture, payload, steps)
        self.assertEqual((observation["result"], observation["error"]["code"]), (None, "invalid_findings"))
        self.assertEqual((row["reward"], row["structure_error"], row["tp"], row["fp"], row["fn"], row["f1"]),
                         (-1.0, code, None, None, None, None))


class Block3Test(AdapterBlock):
    def test_01_one_correct_finding(self):
        a = single_a()
        self.check(a, [finding(a)], 1, 0, 0, 0.98, precision=1, recall=1, f1=1)

    def test_02_empty_findings(self):
        a = single_a()
        self.check(a, [], 0, 0, 1, -1, precision=0, recall=0, f1=0)

    def test_03_empty_truth_empty_findings(self):
        clean = Fixture(_generated("clean", CLEAN_SEED, CLEAN_CONFIG))  # a real clean episode: truth is empty
        self.check(clean, [], 0, 0, 0, 0.98, precision=0, recall=0, f1=1)

    def test_04_empty_truth_one_false_finding(self):
        clean = Fixture(_generated("clean", CLEAN_SEED, CLEAN_CONFIG))
        self.check(clean, [finding(clean)], 0, 1, 0, -1, f1=0)

    def test_05_two_identical_findings(self):
        a = single_a()
        result = self.check(a, [finding(a), finding(a)], 1, 1, 0, Fraction(47, 150),
                            precision=Fraction(1, 2), recall=1, f1=Fraction(2, 3))
        self.assertAlmostEqual(result["return"], 0.3133333333, delta=TOL)
        best = judge_adapter.score_episode(a.dir, self.submit(a, [finding(a)]), 10, journal_path=self.log_path())
        self.assertLess(result["return"], best["return"])  # not above test 1

    def test_06_wrong_type(self):
        a = single_a()
        self.check(a, [finding(a, kind="fx_error")], 0, 1, 1, -1, f1=0)

    def test_07_other_existing_record(self):
        a = single_a()
        self.assertIn(a.R[2], a.known)
        self.check(a, [finding(a, affected=[a.R[2]])], 0, 1, 1, -1, f1=0)

    def test_08_amount_off_by_one_at_tolerance_zero(self):
        a = single_a()
        self.check(a, [finding(a, amount=10001)], 0, 1, 1, -1, f1=0)

    def test_09_other_currency(self):
        a = single_a()
        self.check(a, [finding(a, currency="USD")], 0, 1, 1, -1, f1=0)

    def test_10_only_d1(self):
        a = single_a()
        self.check(a, [finding(a, evidence=a.D[:1])], 0, 1, 1, -1, f1=0)

    def test_11_missing_evidence_id_instead_of_d2(self):
        a = single_a()
        self.assertNotIn(a.U, a.known)
        self.check(a, [finding(a, evidence=[a.D[0], a.U])], 0, 1, 1, -1, f1=0)

    def test_12_mass_flagging_five_records_four_types(self):
        a = single_a()
        every = [finding(a, affected=[record], kind=kind) for record in a.R[:5] for kind in TYPES]
        self.assertEqual(len(every), 20)
        result = self.check(a, every, 1, 19, 0, Fraction(-871, 1050),
                            precision=Fraction(1, 20), recall=1, f1=Fraction(2, 21))
        self.assertAlmostEqual(result["return"], -0.8295238095, delta=TOL)

    def test_13_two_errors_one_found(self):
        b = two_errors()
        self.check(b, [finding(b)], 1, 0, 1, Fraction(47, 150),
                   precision=1, recall=Fraction(1, 2), f1=Fraction(2, 3))

    def test_14_two_errors_one_found_one_false(self):
        b = two_errors()
        self.check(b, [finding(b), finding(b, affected=[b.R[2]])], 1, 1, 1, -0.02,
                   precision=Fraction(1, 2), recall=Fraction(1, 2), f1=Fraction(1, 2))

    def test_15_zero_inspection_steps(self):
        a = single_a()
        self.check(a, [finding(a)], 1, 0, 0, 1, steps=0, f1=1)

    def test_16_fifty_inspection_steps(self):
        a = single_a()
        result = self.check(a, [finding(a)], 1, 0, 0, 0.90, steps=50, f1=1)
        base = judge_adapter.score_episode(a.dir, self.submit(a, [finding(a)]), 10, journal_path=self.log_path())
        self.assertAlmostEqual(base["return"] - result["return"], 0.08, delta=TOL)

    def test_17_hundred_inspection_steps(self):
        a = single_a()  # a formula test: the episode itself would end at 100 exploratory actions without a submit
        self.check(a, [finding(a)], 1, 0, 0, 0.80, steps=100, f1=1)

    def test_18_thirty_one_findings(self):
        a = single_a()
        self.structure(a, self.submit(a, [finding(a)] * 31), "too_many_findings")
        self.check(a, [finding(a)] * 30, 1, 29, 0, formula(1, 29, 0, 10))  # the limit itself is fine

    def test_19_eleven_different_evidence_ids(self):
        a = single_a()
        self.structure(a, self.submit(a, [finding(a, evidence=a.D[:11])]), "too_many_evidence_ids")
        self.check(a, [finding(a, evidence=a.D[:10])], 1, 0, 0, 0.98)  # ten existing ids: a superset, TP

    def test_20_explanation_of_1001_ascii_characters(self):
        a = single_a()
        self.structure(a, self.submit(a, [finding(a, explanation="a" * 1001)]), "explanation_too_long")
        self.check(a, [finding(a, explanation="a" * 1000)], 1, 0, 0, 0.98)


class OpenExamplesTest(AdapterBlock):
    """M01-M12: A with affected_ids {R1, R2}, 10000 KZT, evidence {D1, D2}, tau 0; D3 exists, U does not."""

    def f(self, fixture: Fixture, **kw) -> dict:
        kw.setdefault("affected", fixture.R[:2])
        return finding(fixture, **kw)

    def test_baseline_correct_finding(self):
        a = pair_a()
        self.check(a, [self.f(a)], 1, 0, 0, 0.98)

    def test_M01_affected_ids_repeat_is_the_same_set(self):
        a = pair_a()
        r1, r2 = a.R[:2]
        self.check(a, [self.f(a, affected=[r2, r1, r1])], 1, 0, 0, 0.98)
        # a repeated id inside a list is not a repeated finding; a repeated finding is its own FP
        self.check(a, [self.f(a, affected=[r2, r1, r1]), self.f(a)], 1, 1, 0, Fraction(47, 150))

    def test_M02_extra_existing_unaffected_record(self):
        a = pair_a()
        self.check(a, [self.f(a, affected=a.R[:3])], 0, 1, 1, -1)

    def test_M03_superset_with_existing_available_evidence(self):
        a = pair_a()
        d1, d2, d3 = a.D[:3]
        self.check(a, [self.f(a, evidence=[d2, d1, d3])], 1, 0, 0, 0.98)

    def test_M04_unavailable_id_beside_the_minimal_set(self):
        a = pair_a()
        self.check(a, [self.f(a, evidence=[a.D[0], a.D[1], a.U])], 0, 1, 1, -1)

    def test_M05_raw_limit_before_repeats_collapse(self):
        a = pair_a()
        d1, d2 = a.D[:2]
        self.structure(a, self.submit(a, [self.f(a, evidence=[d1] * 10 + [d2])]), "too_many_evidence_ids")
        self.check(a, [self.f(a, evidence=[d1] * 9 + [d2])], 1, 0, 0, 0.98)  # ten raw elements are fine

    def test_M06_currency_is_exact(self):
        a = pair_a()
        for wrong in ("kzt", " KZT", "KZT "):
            with self.subTest(repr(wrong)):
                self.check(a, [self.f(a, currency=wrong)], 0, 1, 1, -1)

    def test_M07_amount_one_off_with_tau_zero(self):
        a = pair_a()
        self.check(a, [self.f(a, amount=10001)], 0, 1, 1, -1)

    def test_M08_tolerance_one_is_inclusive(self):
        a = pair_a(tolerance=1, name="pair_a_tau1")  # a separate fixture; the other fixtures keep tau 0
        policy = json.loads((pathlib.Path(a.dir) / "public" / "policy.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["amount_tolerance_minor"][T], 1)
        self.check(a, [self.f(a, amount=10001)], 1, 0, 0, 0.98)
        self.check(a, [self.f(a, amount=10002)], 0, 1, 1, -1)

    def test_M09_amount_string_or_bool(self):
        a = pair_a()
        self.structure(a, self.submit(a, [self.f(a, amount="10000")]), "amount_not_int")
        self.structure(a, self.submit(a, [self.f(a, amount=True)]), "amount_not_int")

    def test_negative_amount_is_a_false_positive_not_a_structure_error(self):
        # "claimed_amount_minor must be a JSON integer": -5 is the right type, so it only fails to match (DECISIONS U)
        a = pair_a()
        self.check(a, [self.f(a), self.f(a, amount=-5)], 1, 1, 0, formula(1, 1, 0, 10), env=True)
        self.check(a, [self.f(a, amount=-5)], 0, 1, 1, -1)

    def test_M10_maximum_matching_beats_greedy(self):
        b = pair_ab("d12", "d13", "pair_m10")  # A needs {D1, D2}, B needs {D1, D3}
        d1, d2, d3 = b.D[:3]
        f0 = self.f(b, evidence=[d1, d2, d3])  # fits A and B
        f1 = self.f(b, evidence=[d1, d2])      # fits A only
        for order in ([f0, f1], [f1, f0]):
            self.check(b, order, 2, 0, 0, formula(2, 0, 0, 10))  # F1 -> A, F0 -> B: both found

    def test_M11_both_fit_both_and_order_does_not_matter(self):
        b = pair_ab("d12", "d12", "pair_m11")
        f0, f1 = self.f(b, explanation="first"), self.f(b, explanation="second")
        first = self.check(b, [f0, f1], 2, 0, 0, formula(2, 0, 0, 10))
        second = self.check(b, [f1, f0], 2, 0, 0, formula(2, 0, 0, 10))
        self.assertEqual(first, second)

    def test_M12_two_identical_correct_findings(self):
        a = pair_a()
        self.check(a, [self.f(a), self.f(a)], 1, 1, 0, Fraction(47, 150))  # the reward vector of test 5


class RealEpisodeTest(AdapterBlock):
    """The same arithmetic on UNEDITED generated episodes: the truth is the real injection (`verify` passes)."""

    def real(self):
        two = Fixture(_generated("two_real", TWO_SEED, TWO_CONFIG))
        episode = load_episode(two.dir)
        self.assertEqual(verify(episode), [])
        self.assertEqual(len(episode["private"]["truth"]["issues"]), 2)
        answer = oracle_findings({"episode": episode})["findings"]
        self.assertEqual(len(answer), 2)
        return two, answer

    def test_clean_episode_is_verified_and_has_an_empty_truth(self):
        clean = Fixture(_generated("clean", CLEAN_SEED, CLEAN_CONFIG))
        episode = load_episode(clean.dir)
        self.assertEqual(verify(episode), [])
        self.assertEqual(episode["private"]["truth"]["issues"], [])

    def test_oracle_two_errors(self):
        two, answer = self.real()
        self.check(two, answer, 2, 0, 0, formula(2, 0, 0, 10), f1=1)

    def test_one_of_two_found(self):
        two, answer = self.real()
        self.check(two, answer[:1], 1, 0, 1, Fraction(47, 150))

    def test_one_found_and_one_false(self):
        two, answer = self.real()
        false = dict(answer[0], affected_ids=[two.R[0]], evidence_ids=two.D[:1])
        self.assertNotIn(two.R[0], [i for f in answer for i in f["affected_ids"]])
        self.check(two, answer[:1] + [false], 1, 1, 1, -0.02)

    def test_nothing_found_and_a_repeated_correct_finding(self):
        two, answer = self.real()
        self.check(two, [], 0, 0, 2, -1)
        self.check(two, answer[:1] * 2, 1, 1, 1, -0.02)


if __name__ == "__main__":
    unittest.main()
