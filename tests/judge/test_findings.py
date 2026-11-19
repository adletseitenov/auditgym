"""J1: structure table. Each row: payload change -> passes, or StructureError with this code."""
import unittest

from auditgym.findings import StructureError, parse_findings
from tests.judge.fixture import EPISODE_ID, correct_finding, payload

OK = None


def many(n, **changes):
    return payload(*[correct_finding(**changes) for _ in range(n)])


def with_extra(**top):
    body = payload(correct_finding(note="ignored"))
    body.update(top)
    return body


# (case name, payload, expected error code or OK)
CASES = [
    # --- passes ---
    ("valid finding", payload(correct_finding()), OK),
    ("extra keys at both levels are ignored", with_extra(debug="x"), OK),
    ("empty findings list", payload(), OK),
    ("unknown ids are a matcher question, not structure", payload(correct_finding(affected_ids=["nope"])), OK),
    ("unknown currency is a matcher question", payload(correct_finding(currency="XYZ")), OK),
    ("amount zero", payload(correct_finding(claimed_amount_minor=0)), OK),
    ("empty affected_ids", payload(correct_finding(affected_ids=[])), OK),
    ("empty evidence_ids", payload(correct_finding(evidence_ids=[])), OK),
    ("exactly 30 findings", many(30), OK),
    ("exactly 10 evidence ids", payload(correct_finding(evidence_ids=[f"e{i}" for i in range(10)])), OK),
    ("explanation of exactly 1000 characters", payload(correct_finding(explanation="a" * 1000)), OK),
    ("1000 Cyrillic characters (2000 bytes)", payload(correct_finding(explanation="ж" * 1000)), OK),
    ("empty explanation", payload(correct_finding(explanation="")), OK),
    # --- reward -1 ---
    ("payload is a list", [correct_finding()], "not_object"),
    ("payload is a string", '{"episode_id": "ep_tiny0001", "findings": []}', "not_object"),
    ("payload is null", None, "not_object"),
    ("no episode_id", {"findings": []}, "missing_field"),
    ("no findings", {"episode_id": EPISODE_ID}, "missing_field"),
    ("episode_id of another episode", payload(episode_id="ep_other"), "wrong_episode_id"),
    ("episode_id is not a string", {"episode_id": 1, "findings": []}, "wrong_episode_id"),
    ("findings is an object", {"episode_id": EPISODE_ID, "findings": {}}, "findings_not_list"),
    ("31 findings", many(31), "too_many_findings"),
    ("finding is a string", payload("duplicate_invoice"), "finding_not_object"),
    ("finding without explanation",
     payload({k: v for k, v in correct_finding().items() if k != "explanation"}), "missing_field"),
    ("finding without evidence_ids",
     payload({k: v for k, v in correct_finding().items() if k != "evidence_ids"}), "missing_field"),
    ("unknown type", payload(correct_finding(type="fraud")), "unknown_type"),
    ("type in a list", payload(correct_finding(type=["duplicate_invoice"])), "unknown_type"),
    ("amount true", payload(correct_finding(claimed_amount_minor=True)), "amount_not_int"),
    ("amount 100.0", payload(correct_finding(claimed_amount_minor=100.0)), "amount_not_int"),
    ("amount as string", payload(correct_finding(claimed_amount_minor="5000000")), "amount_not_int"),
    ("amount null", payload(correct_finding(claimed_amount_minor=None)), "amount_not_int"),
    ("negative amount", payload(correct_finding(claimed_amount_minor=-5000000)), "amount_negative"),
    ("NaN amount is not JSON", payload(correct_finding(claimed_amount_minor=float("nan"))), "not_json"),
    ("currency is a number", payload(correct_finding(currency=398)), "currency_not_string"),
    ("11 evidence ids", payload(correct_finding(evidence_ids=[f"e{i}" for i in range(11)])), "too_many_evidence_ids"),
    ("11 evidence ids with repeats are counted as sent",
     payload(correct_finding(evidence_ids=["a"] * 11)), "too_many_evidence_ids"),
    ("evidence id is a number", payload(correct_finding(evidence_ids=["k7m2qa", 5])), "bad_evidence_ids"),
    ("evidence_ids is a string", payload(correct_finding(evidence_ids="k7m2qa")), "bad_evidence_ids"),
    ("affected_ids is a string", payload(correct_finding(affected_ids="k7m2qa")), "bad_affected_ids"),
    ("affected id is null", payload(correct_finding(affected_ids=[None])), "bad_affected_ids"),
    ("explanation of 1001 characters", payload(correct_finding(explanation="a" * 1001)), "explanation_too_long"),
    ("explanation is null", payload(correct_finding(explanation=None)), "explanation_not_string"),
    ("payload above 64 KiB", payload(correct_finding(padding="x" * 70000)), "payload_too_large"),
    ("value that is not JSON", payload(correct_finding(extra={1, 2})), "not_json"),
]


class StructureTableTest(unittest.TestCase):
    def test_table(self):
        self.assertGreaterEqual(len(CASES), 25)
        for name, body, expected in CASES:
            with self.subTest(name):
                if expected is OK:
                    parse_findings(body, EPISODE_ID)
                else:
                    with self.assertRaises(StructureError) as caught:
                        parse_findings(body, EPISODE_ID)
                    self.assertEqual(caught.exception.code, expected)


class NormalizationTest(unittest.TestCase):
    def test_ids_are_deduplicated_and_sorted(self):
        body = payload(correct_finding(affected_ids=["z5c1ty", "k7m2qa", "z5c1ty"],
                                       evidence_ids=["z5c1ty", "k7m2qa", "k7m2qa"]))
        [finding] = parse_findings(body, EPISODE_ID)
        self.assertEqual(finding["affected_ids"], ["k7m2qa", "z5c1ty"])
        self.assertEqual(finding["evidence_ids"], ["k7m2qa", "z5c1ty"])

    def test_extra_keys_are_dropped_and_order_is_kept(self):
        body = payload(correct_finding(note="x"), correct_finding(type="fx_error"))
        findings = parse_findings(body, EPISODE_ID)
        self.assertEqual([f["type"] for f in findings], ["duplicate_invoice", "fx_error"])
        self.assertNotIn("note", findings[0])

    def test_input_is_not_mutated(self):
        body = payload(correct_finding(affected_ids=["z5c1ty", "k7m2qa"]))
        parse_findings(body, EPISODE_ID)
        self.assertEqual(body["findings"][0]["affected_ids"], ["z5c1ty", "k7m2qa"])


if __name__ == "__main__":
    unittest.main()
