"""J4 fuzz: thousands of JSON-compatible junk payloads never crash the judge.

Uses auditgym.rng.Rng (never `random`). Two sources: fully random JSON values, and near-valid
payloads (the oracle answer with one or more random edits), which reach deeper code paths.
"""
import copy
import json
import unittest

from auditgym.findings import StructureError, message_size, parse_findings
from auditgym.judge import judge
from auditgym.rng import Rng
from tests.judge import helpers

FIELDS = ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation",
          "episode_id", "findings", "x")
STRINGS = ("", "a00", "b00", "r00", "f01", "duplicate_invoice", "fx_error", "KZT", "kzt", "ep_hand0001",
           "ж", "\U0001F600", "\ud800", "\x00", "2026-10-01|USD|KZT", "a" * 1001)
FLOATS = (0.0, -0.0, 1.5, 1000.0, 1e308, -1e-308, float("nan"), float("inf"), float("-inf"))
INTS = (0, 1, -1, 1000, 2 ** 63, -(2 ** 63), 10 ** 30, 10 ** 5000)


def junk(rng: Rng, depth: int = 0):
    kind = rng.below(9 if depth < 4 else 6)
    if kind == 0:
        return None
    if kind == 1:
        return rng.chance(1, 2)
    if kind == 2:
        return rng.choice(INTS) if rng.chance(1, 3) else rng.randint(-5, 5000)
    if kind == 3:
        return rng.choice(FLOATS)
    if kind in (4, 5):
        return rng.choice(STRINGS) if rng.chance(2, 3) else rng.token(rng.below(8))
    if kind in (6, 7):
        return [junk(rng, depth + 1) for _ in range(rng.below(5))]
    return {rng.choice(FIELDS): junk(rng, depth + 1) for _ in range(rng.below(5))}


def containers(value, out):
    if isinstance(value, (list, dict)):
        out.append(value)
        for child in (value if isinstance(value, list) else value.values()):
            containers(child, out)
    return out


def near_valid(rng: Rng, ep: dict):
    payload = copy.deepcopy(helpers.submit(ep, helpers.oracle(ep) + [helpers.false_accusation(1)]))
    for _ in range(1 + rng.below(3)):
        target = rng.choice(containers(payload, []))
        action = rng.below(4)
        if isinstance(target, dict):
            keys = sorted(target)
            if action == 0 and keys:
                del target[rng.choice(keys)]
            elif action == 1 and keys:
                target[rng.choice(keys)] = junk(rng)
            else:
                target[rng.choice(FIELDS)] = junk(rng)
        else:
            if action == 0 and target:
                del target[rng.below(len(target))]
            elif action == 1 and target:
                target[rng.below(len(target))] = junk(rng)
            elif action == 2 and target:
                target.append(copy.deepcopy(rng.choice(target)))
            else:
                target.append(junk(rng))
    return payload


class FuzzTest(unittest.TestCase):
    def test_judge_never_raises_and_reward_in_range(self):
        rng = Rng(2026, "judge-fuzz")
        episodes = [helpers.episode(n) for n in (0, 1, 3, 12)]
        codes = set()
        for i in range(10000):
            ep = episodes[i % 4]
            payload = near_valid(rng, ep) if i % 2 else junk(rng)
            if rng.chance(1, 10) and isinstance(payload, dict):
                payload["episode_id"] = ep["public"]["manifest"]["episode_id"]
            steps = rng.randint(0, 100)
            try:
                result = judge(ep, payload, steps)
            except Exception as error:  # pragma: no cover - the assertion is the point
                self.fail(f"judge raised {error!r} on payload {str(payload)[:200]}")
            self.assertTrue(-1.0 <= result["reward"] <= 1.0)
            self.assertTrue(0 <= result["tp"] <= len(ep["private"]["truth"]["issues"]))
            codes.add(result["structure_error"])
            try:
                parse_findings(payload, ep["public"]["manifest"]["episode_id"])
                parsed_code = None
            except StructureError as error:
                parsed_code = error.code
            # None is the truncation signal for judge (empty answer), not a submitted payload.
            self.assertEqual(result["structure_error"], None if payload is None else parsed_code)
        self.assertIn(None, codes)
        self.assertGreater(len(codes), 6)  # the fuzz reaches many distinct checks

    def test_size_matches_json_on_finite_payloads(self):
        rng = Rng(7, "size-fuzz")
        checked = 0
        for _ in range(1500):
            payload = junk(rng)
            try:
                text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            except ValueError:
                continue  # non-finite or an int above the str() limit
            self.assertEqual(message_size(payload), len(text.encode("utf-8", "surrogatepass")))
            checked += 1
        self.assertGreater(checked, 1000)


if __name__ == "__main__":
    unittest.main()
