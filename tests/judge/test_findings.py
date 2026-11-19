"""J1: structure check and normalization (docs/CONTRACTS.md §6). Table-driven."""
import json
import unittest

from auditgym.findings import MAX_MESSAGE_BYTES, StructureError, message_size, parse_findings

EP = "ep_x"
GOOD = {"type": "duplicate_invoice", "affected_ids": ["b", "a"], "claimed_amount_minor": 100,
        "currency": "usd", "evidence_ids": ["a", "b"], "explanation": "same items"}


def finding(**changes):
    item = dict(GOOD)
    item.update(changes)
    return item


def without(field):
    return {k: v for k, v in GOOD.items() if k != field}


def msg(*findings, **top):
    payload = {"episode_id": EP, "findings": list(findings)}
    payload.update(top)
    return payload


def compact_bytes(payload):
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
               .encode("utf-8", "surrogatepass"))


# (name, payload, expected code or None for valid)
CASES = [
    ("valid single finding", msg(GOOD), None),
    ("valid empty findings", msg(), None),
    ("not an object: list", [GOOD], "not_object"),
    ("not an object: string", "findings", "not_object"),
    ("not an object: None-like 0", 0, "not_object"),
    ("missing episode_id", {"findings": []}, "missing_field"),
    ("missing findings", {"episode_id": EP}, "missing_field"),
    ("findings not a list", {"episode_id": EP, "findings": {"a": GOOD}}, "findings_not_list"),
    ("30 findings", msg(*[GOOD] * 30), None),
    ("31 findings", msg(*[GOOD] * 31), "too_many_findings"),
    ("finding not an object", msg("dup"), "finding_not_object"),
    *[(f"missing {f}", msg(without(f)), "missing_field")
      for f in ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation")],
    ("unknown type", msg(finding(type="fraud")), "unknown_type"),
    ("type not a string", msg(finding(type=["duplicate_invoice"])), "unknown_type"),
    ("type case differs", msg(finding(type="Duplicate_Invoice")), "unknown_type"),
    ("wrong episode_id", msg(GOOD, episode_id="ep_y"), "wrong_episode_id"),
    ("episode_id not a string", msg(GOOD, episode_id=1), "wrong_episode_id"),
    ("amount bool", msg(finding(claimed_amount_minor=True)), "amount_not_int"),
    ("amount float 100.0", msg(finding(claimed_amount_minor=100.0)), "amount_not_int"),
    ("amount string", msg(finding(claimed_amount_minor="100")), "amount_not_int"),
    ("amount None", msg(finding(claimed_amount_minor=None)), "amount_not_int"),
    ("amount negative", msg(finding(claimed_amount_minor=-1)), "negative_amount"),
    ("amount zero", msg(finding(claimed_amount_minor=0)), None),
    ("amount 10**30", msg(finding(claimed_amount_minor=10 ** 30)), None),
    ("amount 10**70000 (too large)", msg(finding(claimed_amount_minor=10 ** 70000)), "message_too_large"),
    ("amount NaN", msg(finding(claimed_amount_minor=float("nan"))), "non_finite"),
    ("Infinity in an extra key", msg(GOOD, extra=float("inf")), "non_finite"),
    ("affected_ids not a list", msg(finding(affected_ids="a")), "ids_not_list"),
    ("evidence_ids a dict", msg(finding(evidence_ids={"a": 1})), "ids_not_list"),
    ("affected id not a string", msg(finding(affected_ids=["a", 1])), "id_not_string"),
    ("evidence id None", msg(finding(evidence_ids=[None])), "id_not_string"),
    ("10 evidence ids", msg(finding(evidence_ids=[str(i) for i in range(10)])), None),
    ("11 evidence ids", msg(finding(evidence_ids=[str(i) for i in range(11)])), "too_many_evidence_ids"),
    ("11 evidence ids with repeats", msg(finding(evidence_ids=["a"] * 11)), "too_many_evidence_ids"),
    ("empty affected and evidence", msg(finding(affected_ids=[], evidence_ids=[])), None),
    ("currency not a string", msg(finding(currency=840)), "currency_not_string"),
    ("explanation 1000 code points", msg(finding(explanation="ж" * 1000)), None),
    ("explanation 1001 code points", msg(finding(explanation="ж" * 1001)), "explanation_too_long"),
    ("explanation 1000 astral chars", msg(finding(explanation="\U0001F600" * 1000)), None),
    ("explanation 1001 ascii", msg(finding(explanation="a" * 1001)), "explanation_too_long"),
    ("explanation empty", msg(finding(explanation="")), None),
    ("explanation not a string", msg(finding(explanation=None)), "explanation_not_string"),
    ("extra keys ignored", msg(finding(verdict="tp"), note={"x": [1, None, True]}), None),
    ("payload over 64 KiB", msg(GOOD, pad="x" * MAX_MESSAGE_BYTES), "message_too_large"),
    ("non-string dict key", msg(GOOD, extra={1: "a"}), "not_json"),
    ("set value", msg(GOOD, extra={"a"}), "not_json"),
    ("bytes value", msg(GOOD, extra=b"a"), "not_json"),
]


class StructureTableTest(unittest.TestCase):
    def test_table(self):
        for name, payload, code in CASES:
            with self.subTest(name):
                if code is None:
                    parse_findings(payload, EP)
                else:
                    with self.assertRaises(StructureError) as caught:
                        parse_findings(payload, EP)
                    self.assertEqual(caught.exception.code, code)


class NormalizationTest(unittest.TestCase):
    def test_ids_sorted_unique_and_currency_upper(self):
        [item] = parse_findings(msg(finding(affected_ids=["b", "a", "b"], evidence_ids=["b", "a", "a"])), EP)
        self.assertEqual(item["affected_ids"], ["a", "b"])
        self.assertEqual(item["evidence_ids"], ["a", "b"])
        self.assertEqual(item["currency"], "USD")
        self.assertNotIn("verdict", parse_findings(msg(finding(verdict=1)), EP)[0])

    def test_upper_case_is_ascii_only(self):
        [item] = parse_findings(msg(finding(currency="uſd")), EP)
        self.assertEqual(item["currency"], "UſD")

    def test_input_is_not_mutated(self):
        payload = msg(finding(affected_ids=["b", "a"]))
        before = json.dumps(payload)
        parse_findings(payload, EP)
        self.assertEqual(json.dumps(payload), before)


class MessageSizeTest(unittest.TestCase):
    def test_equals_compact_utf8_json(self):
        samples = [msg(GOOD), msg(finding(explanation="ж\n\"\\\x01\x7f \U0001F600\ud800")),
                   {"a": [1, -2, 3.5, -0.0, 1e300, None, True, False, "", [], {}]},
                   msg(finding(claimed_amount_minor=10 ** 4000))]
        for payload in samples:
            with self.subTest(payload=str(payload)[:40]):
                self.assertEqual(message_size(payload), compact_bytes(payload))

    def test_exact_small(self):
        for payload in ([], {}, "ж", [1, [2, {"k": None}]], {"a": "\x00"}):
            self.assertEqual(message_size(payload), compact_bytes(payload))

    def test_huge_int_size_without_str(self):
        value = 10 ** 4400  # above the 4300-digit str() limit of Python 3.11
        self.assertEqual(message_size([value]), 2 + 4401)

    def test_boundary_64_kib(self):
        base = message_size(msg(GOOD, pad=""))
        fits = msg(GOOD, pad="x" * (MAX_MESSAGE_BYTES - base))
        self.assertEqual(message_size(fits), MAX_MESSAGE_BYTES)
        parse_findings(fits, EP)
        over = msg(GOOD, pad="x" * (MAX_MESSAGE_BYTES - base + 1))
        with self.assertRaises(StructureError):
            parse_findings(over, EP)

    def test_cycle_and_depth_do_not_crash(self):
        loop = []
        loop.append(loop)
        with self.assertRaises(StructureError):
            parse_findings(msg(GOOD, pad=loop), EP)
        deep = []
        for _ in range(100_000):
            deep = [deep]
        with self.assertRaises(StructureError):  # 200 000 brackets > 64 KiB
            parse_findings(msg(GOOD, pad=deep), EP)
        shallow = []
        for _ in range(10_000):
            shallow = [shallow]
        parse_findings(msg(GOOD, pad=shallow), EP)


if __name__ == "__main__":
    unittest.main()
