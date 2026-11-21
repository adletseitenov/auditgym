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
      for f in ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids")],
    ("missing explanation is the empty string (R5)", msg(without("explanation")), None),
    ("unknown type string is well-formed (an FP, R)", msg(finding(type="fraud")), None),
    ("type not a string", msg(finding(type=["duplicate_invoice"])), "type_not_string"),
    ("type case and spaces are kept as they are (an FP, R)", msg(finding(type="  Duplicate_Invoice\t")), None),
    ("wrong episode_id", msg(GOOD, episode_id="ep_y"), "wrong_episode_id"),
    ("episode_id not a string", msg(GOOD, episode_id=1), "wrong_episode_id"),
    ("amount bool", msg(finding(claimed_amount_minor=True)), "amount_not_int"),
    ("amount float 100.0 is not a JSON integer (R)", msg(finding(claimed_amount_minor=100.0)), "amount_not_int"),
    ("amount float 100.5", msg(finding(claimed_amount_minor=100.5)), "amount_not_int"),
    ("amount float -100.0", msg(finding(claimed_amount_minor=-100.0)), "amount_not_int"),
    ("amount Infinity", msg(finding(claimed_amount_minor=float("inf"))), "non_finite"),
    ("amount string", msg(finding(claimed_amount_minor="100")), "amount_not_int"),
    ("amount None", msg(finding(claimed_amount_minor=None)), "amount_not_int"),
    ("amount negative", msg(finding(claimed_amount_minor=-1)), "negative_amount"),
    ("amount zero", msg(finding(claimed_amount_minor=0)), None),
    ("amount 10**30", msg(finding(claimed_amount_minor=10 ** 30)), None),
    ("amount 10**70000 fits in 256 KiB", msg(finding(claimed_amount_minor=10 ** 70000)), None),
    ("amount 10**300000 (too large)", msg(finding(claimed_amount_minor=10 ** 300000)), "message_too_large"),
    ("amount NaN", msg(finding(claimed_amount_minor=float("nan"))), "non_finite"),
    ("Infinity in an extra key", msg(GOOD, extra=float("inf")), "non_finite"),
    ("affected_ids not a list", msg(finding(affected_ids="a")), "ids_not_list"),
    ("evidence_ids a dict", msg(finding(evidence_ids={"a": 1})), "ids_not_list"),
    ("affected id not a string", msg(finding(affected_ids=["a", 1])), "id_not_string"),
    ("evidence id None", msg(finding(evidence_ids=[None])), "id_not_string"),
    ("10 evidence ids", msg(finding(evidence_ids=[str(i) for i in range(10)])), None),
    ("11 evidence ids", msg(finding(evidence_ids=[str(i) for i in range(11)])), "too_many_evidence_ids"),
    ("11 different evidence ids among repeats", msg(finding(evidence_ids=[str(i % 11) for i in range(25)])),
     "too_many_evidence_ids"),
    ("the limit counts the raw list: 25 entries, 10 different (R, M05)",
     msg(finding(evidence_ids=[str(i % 10) for i in range(25)])), "too_many_evidence_ids"),
    ("11 entries with 10 repeats of one id (M05)", msg(finding(evidence_ids=["d1"] * 10 + ["d2"])),
     "too_many_evidence_ids"),
    ("10 entries with repeats are fine", msg(finding(evidence_ids=["d1"] * 9 + ["d2"])), None),
    ("a repeated affected id is well-formed and is the set (R, M01)",
     msg(finding(affected_ids=["a", "b", "a"])), None),
    ("empty affected and evidence", msg(finding(affected_ids=[], evidence_ids=[])), None),
    ("currency not a string", msg(finding(currency=840)), "currency_not_string"),
    ("explanation 1000 code points", msg(finding(explanation="ж" * 1000)), None),
    ("explanation 1001 code points", msg(finding(explanation="ж" * 1001)), "explanation_too_long"),
    ("explanation 1000 astral chars", msg(finding(explanation="\U0001F600" * 1000)), None),
    ("explanation 1001 ascii", msg(finding(explanation="a" * 1001)), "explanation_too_long"),
    ("explanation empty", msg(finding(explanation="")), None),
    ("explanation not a string", msg(finding(explanation=None)), "explanation_not_string"),
    ("extra keys ignored", msg(finding(verdict="tp"), note={"x": [1, None, True]}), None),
    ("payload over 256 KiB", msg(GOOD, pad="x" * MAX_MESSAGE_BYTES), "message_too_large"),
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
    def test_repeated_affected_id_collapses_to_the_set(self):
        [item] = parse_findings(msg(finding(affected_ids=["b", "a", "b"])), EP)
        self.assertEqual(item["affected_ids"], ["a", "b"])

    def test_currency_and_type_are_kept_exactly(self):
        [item] = parse_findings(msg(finding(currency=" usd\n", type=" FX_Error ")), EP)
        self.assertEqual((item["currency"], item["type"]), (" usd\n", " FX_Error "))

    def test_integral_float_amount_is_refused(self):
        with self.assertRaises(StructureError) as caught:
            parse_findings(msg(finding(claimed_amount_minor=125000.0)), EP)
        self.assertEqual(caught.exception.code, "amount_not_int")

    def test_explanation_defaults_to_empty(self):
        [item] = parse_findings(msg(without("explanation")), EP)
        self.assertEqual(item["explanation"], "")

    def test_ids_sorted_unique_and_currency_kept(self):
        [item] = parse_findings(msg(finding(affected_ids=["b", "a"], evidence_ids=["b", "a", "a"])), EP)
        self.assertEqual(item["affected_ids"], ["a", "b"])
        self.assertEqual(item["evidence_ids"], ["a", "b"])
        self.assertEqual(item["currency"], "usd")
        self.assertNotIn("verdict", parse_findings(msg(finding(verdict=1)), EP)[0])

    def test_currency_is_not_case_folded(self):
        [item] = parse_findings(msg(finding(currency="uſd")), EP)
        self.assertEqual(item["currency"], "uſd")

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

    def test_limit_is_256_kib(self):
        self.assertEqual(MAX_MESSAGE_BYTES, 262144)
        base = message_size(msg(GOOD, pad=""))
        parse_findings(msg(GOOD, pad="x" * (262144 - base)), EP)
        with self.assertRaises(StructureError) as caught:
            parse_findings(msg(GOOD, pad="x" * (262144 - base + 1)), EP)
        self.assertEqual(caught.exception.code, "message_too_large")

    def test_the_case_maximum_fits_in_cyrillic(self):
        """R2: 30 findings x 10 evidence ids x 1000 characters is 68 KB and must not be -1."""
        big = finding(explanation="ж" * 1000, evidence_ids=[f"e{i:02d}" for i in range(10)])
        answer = msg(*[big] * 30)
        self.assertGreater(message_size(answer), 65536)
        self.assertEqual(len(parse_findings(answer, EP)), 30)

    def test_huge_int_exact_digits(self):
        for exponent in (9_999, 10_000, 100_000, 261_000):
            value = 10 ** exponent
            self.assertEqual(message_size([value]), 2 + exponent + 1, exponent)
            self.assertEqual(message_size([value - 1]), 2 + exponent, exponent)

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
        for _ in range(150_000):
            deep = [deep]
        with self.assertRaises(StructureError):  # 300 000 brackets > 256 KiB
            parse_findings(msg(GOOD, pad=deep), EP)
        shallow = []
        for _ in range(10_000):
            shallow = [shallow]
        parse_findings(msg(GOOD, pad=shallow), EP)


class RepeatedAffectedIdIsTheSetTest(unittest.TestCase):
    """R (official M01): affected_ids is a set; a repeat inside the list changes nothing."""

    @staticmethod
    def with_repeat(item):
        return dict(item, affected_ids=item["affected_ids"] + item["affected_ids"][:1])

    def test_repeats_inside_the_list_are_still_true_positives(self):
        from auditgym.judge import judge
        from tests.judge import helpers

        ep = helpers.episode(4)
        good = helpers.oracle(ep)
        correct = judge(ep, helpers.submit(ep, good), 0)
        repeated = judge(ep, helpers.submit(ep, [self.with_repeat(i) for i in good]), 0)
        self.assertEqual(repeated, correct)
        self.assertEqual((repeated["tp"], repeated["fp"], repeated["fn"]), (4, 0, 0))

    def test_a_repeated_finding_is_not_a_repeated_id(self):
        from auditgym.judge import judge
        from tests.judge import helpers

        ep = helpers.episode(1)
        good = helpers.oracle(ep)
        result = judge(ep, helpers.submit(ep, good + [self.with_repeat(good[0])]), 0)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))

    def test_a_repeat_next_to_a_stranger_is_still_a_wrong_set(self):
        from auditgym.judge import judge
        from tests.judge import helpers

        ep = helpers.episode(1)
        good = helpers.oracle(ep)
        wrong = dict(good[0], affected_ids=good[0]["affected_ids"] + good[0]["affected_ids"][:1] + ["f00"])
        result = judge(ep, helpers.submit(ep, [wrong]), 0)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

    def test_the_findings_limit_counts_findings_not_ids(self):
        from auditgym.judge import judge
        from tests.judge import helpers

        ep = helpers.episode(1)
        good = helpers.oracle(ep)
        bad = self.with_repeat(good[0])
        fits = judge(ep, helpers.submit(ep, good + [bad] * 29), 0)
        over = judge(ep, helpers.submit(ep, good + [bad] * 30), 0)
        self.assertIsNone(fits["structure_error"])
        self.assertEqual((fits["tp"], fits["fp"]), (1, 29))
        self.assertEqual(over["structure_error"], "too_many_findings")

    def test_other_structure_rules_still_apply_next_to_a_repeat(self):
        for bad, code in ((dict(GOOD, claimed_amount_minor=-1), "negative_amount"),
                          (dict(GOOD, explanation="x" * 1001), "explanation_too_long"),
                          (dict(GOOD, affected_ids=["a", "a", 1]), "id_not_string")):
            with self.subTest(code), self.assertRaises(StructureError) as caught:
                parse_findings(msg(finding(affected_ids=["a", "a"]), bad), EP)
            self.assertEqual(caught.exception.code, code)


if __name__ == "__main__":
    unittest.main()
