"""task.answer_rules against the judge and the truth (docs/DECISIONS.md, "Правила ответа в первом наблюдении").

The policy below reads ONLY `answer_rules` (the roles and the sets of each type) and the open records; the
expected ids, amounts and currencies it takes from the truth, because choosing them is the audit itself.
It never imports the generator's operators or the solver.
"""
import datetime as dt
import json
import pathlib
import unittest

from auditgym.actions import ERROR_TYPES, build_task
from auditgym.generate import generate
from auditgym.judge import judge
from auditgym.storage import load_episode
from tools.fuzz import grid

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RULES = build_task({})["answer_rules"]


def _zone(offset: str) -> dt.timezone:
    sign = -1 if offset[0] == "-" else 1
    return dt.timezone(sign * dt.timedelta(hours=int(offset[1:3]), minutes=int(offset[4:6])))


class RulesPolicy:
    """Builds the evidence of a finding by the text of the rules: a role token is looked up by its description."""

    def __init__(self, public: dict, rules: dict) -> None:
        self.public, self.rules = public, rules
        policy = public["policy"]
        self.zone, self.base = _zone(policy["timezone"]), policy["base_currency"]
        self.invoices = {d["doc_id"]: d for d in public["invoices"]}
        self.payments = {d["doc_id"]: d for d in public["payments"]}
        self.fx = {f"{r['date']}|{r['source_currency']}|{r['target_currency']}" for r in public["fx"]}

    def _rows(self, account, side, invoice_id, payment_id, with_invoice=False):
        return [r["row_id"] for r in self.public["ledger"]
                if r["account"] == account and r[side] > 0 and r["payment_id"] == payment_id
                and (r["invoice_id"] == invoice_id if (invoice_id or with_invoice) else True)]

    def role(self, name: str, doc_id: str, invoice_doc: str | None = None):
        """The id of a role for a document, or None when the open records have no such thing."""
        if name == "doc":
            return doc_id
        if name == "rate_key":
            if doc_id in self.invoices:
                moment, currency = self.invoices[doc_id]["issued_at"], self.invoices[doc_id]["currency"]
            else:
                moment, currency = self.payments[doc_id]["timestamp"], self.payments[doc_id]["currency"]
            date = dt.datetime.fromisoformat(moment).astimezone(self.zone).date().isoformat()
            key = f"{date}|{currency}|{self.base}"
            return key if key in self.fx else None
        if name in ("receivable_debit", "revenue_credit"):
            account, side = ("receivable", "debit_minor") if name == "receivable_debit" else ("revenue", "credit_minor")
            found = self._rows(account, side, self.invoices[doc_id]["invoice_id"] if doc_id in self.invoices else "-", None,
                               with_invoice=True)
            return found[0] if len(found) == 1 else None
        payment_id = self.payments[doc_id]["payment_id"] if doc_id in self.payments else None
        if payment_id is None:
            return None
        if name == "cash_row":
            found = [r["row_id"] for r in self.public["ledger"] if r["account"] == "cash" and r["payment_id"] == payment_id]
        elif name == "unallocated_row":
            found = self._rows("unallocated", "credit_minor", None, payment_id)
        elif name == "receivable_credit":
            invoice_id = self.invoices[invoice_doc]["invoice_id"]
            found = self._rows("receivable", "credit_minor", invoice_id, payment_id)
        else:
            raise AssertionError(f"role without a description: {name}")
        return found[0] if len(found) == 1 else None

    def subjects(self, issue_type: str, affected: list[str]) -> dict:
        if issue_type == "duplicate_invoice":
            first, second = sorted(affected)
            return {"A": first, "B": second}
        if issue_type == "allocation_gap":
            return {"payment": next(i for i in affected if i in self.payments),
                    "invoice": next(i for i in affected if i in self.invoices)}
        return {"document": affected[0]}

    def alternatives(self, issue_type: str, affected: list[str]) -> list[list[str]]:
        """Every set of the rules that the open records can fill."""
        subjects = self.subjects(issue_type, affected)
        other = subjects.get("invoice")
        sets = []
        for tokens in self.rules["types"][issue_type]["evidence"]["one_of"]:
            ids = []
            for token in tokens:
                name, subject = token[:-1].split("(")
                ids.append(self.role(name, subjects[subject], other))
            if None not in ids:
                sets.append(ids)
        return sets

    def answer(self, truth: dict) -> list[dict]:
        findings = []
        for issue in truth["issues"]:
            sets = self.alternatives(issue["type"], issue["affected_ids"])
            assert sets, f"no evidence set can be built for {issue['type']}"
            findings.append({"type": issue["type"], "affected_ids": issue["affected_ids"],
                             "claimed_amount_minor": issue["expected_amount_minor"], "currency": issue["currency"],
                             "evidence_ids": sets[0], "explanation": "by the rules"})
        return findings


def _reward(episode: dict, findings: list) -> dict:
    return judge(episode, {"episode_id": episode["public"]["manifest"]["episode_id"], "findings": findings}, 0)


def _generated() -> list:
    episodes = []
    for index, (name, config) in enumerate(grid()):
        for seed in range(3):
            if len(episodes) < 50:
                episodes.append((f"{name}/{seed}", generate(1000 * seed + index + 7, config)))
    return episodes


class AnswerRulesPolicyTest(unittest.TestCase):
    def check(self, label: str, episode: dict) -> None:
        policy = RulesPolicy(episode["public"], RULES)
        truth = episode["private"]["truth"]
        # the sets the rules describe are exactly the sets of the truth, not a subset
        for issue in truth["issues"]:
            built = {frozenset(s) for s in policy.alternatives(issue["type"], issue["affected_ids"])}
            self.assertEqual(built, {frozenset(s) for s in issue["evidence_sets"]}, f"{label} {issue['type']}")
        result = _reward(episode, policy.answer(truth))
        self.assertEqual((result["reward"], result["fp"], result["fn"]), (1.0, 0, 0), label)
        self.assertEqual(result["tp"], len(truth["issues"]), label)

    def test_twenty_released_episodes_train_and_dev(self):
        count = 0
        for split in ("train", "dev"):
            for path in sorted((ROOT / "episodes" / split).iterdir()):
                self.check(f"{split}/{path.name}", load_episode(path))
                count += 1
        self.assertEqual(count, 20)

    def test_fifty_generated_episodes_of_different_configs_with_i1_and_i2(self):
        episodes = _generated()
        self.assertEqual(len(episodes), 50)
        interactions = {ep["private"]["provenance"]["config"]["interaction"] for _, ep in episodes}
        self.assertTrue({"I1", "I2"} <= interactions)
        for label, episode in episodes:
            self.check(label, episode)

    def test_every_error_type_is_covered(self):
        seen = set()
        for _, episode in _generated():
            seen.update(issue["type"] for issue in episode["private"]["truth"]["issues"])
        self.assertEqual(seen, set(ERROR_TYPES))


class AnswerRulesAreNeededTest(unittest.TestCase):
    def episode_with(self, error_type: str) -> dict:
        for _, episode in _generated():
            if any(i["type"] == error_type for i in episode["private"]["truth"]["issues"]):
                return episode
        raise AssertionError(error_type)

    def test_fx_error_without_the_rate_key_is_a_false_positive(self):
        for _, episode in _generated():
            policy = RulesPolicy(episode["public"], RULES)
            findings = [f for f in policy.answer(episode["private"]["truth"]) if f["type"] == "fx_error"]
            if not findings:
                continue
            for finding in findings:
                self.assertTrue(any("|" in i for i in finding["evidence_ids"]))
                finding["evidence_ids"] = [i for i in finding["evidence_ids"] if "|" not in i]
            result = _reward(episode, findings)
            self.assertEqual((result["tp"], result["fp"]), (0, len(findings)))
            self.assertLess(result["reward"], 1.0)

    def test_the_rules_say_the_rate_key_is_required_in_every_fx_error_set(self):
        sets = RULES["types"]["fx_error"]["evidence"]["one_of"]
        self.assertTrue(sets and all("rate_key(document)" in tokens for tokens in sets))
        self.assertIn("FP", RULES["types"]["fx_error"]["evidence"]["subjects"])

    def test_a_gap_without_the_unallocated_row_is_a_false_positive(self):
        episode = self.episode_with("allocation_gap")
        policy = RulesPolicy(episode["public"], RULES)
        finding = next(f for f in policy.answer(episode["private"]["truth"]) if f["type"] == "allocation_gap")
        unallocated = {r["row_id"] for r in episode["public"]["ledger"] if r["account"] == "unallocated"}
        finding["evidence_ids"] = [i for i in finding["evidence_ids"] if i not in unallocated]
        self.assertEqual(_reward(episode, [finding])["tp"], 0)

    def test_extra_existing_ids_are_allowed_and_a_missing_id_is_not(self):
        episode = self.episode_with("period_cutoff")
        policy = RulesPolicy(episode["public"], RULES)
        truth = episode["private"]["truth"]
        extra = episode["public"]["ledger"][0]["row_id"]
        full = policy.answer(truth)
        for finding in full:
            finding["evidence_ids"] = [*finding["evidence_ids"], extra]
        self.assertEqual(_reward(episode, full)["fp"], 0)
        for finding in full:
            finding["evidence_ids"] = finding["evidence_ids"][:-1] + ["nonexistent-id"]
        self.assertEqual(_reward(episode, full)["tp"], 0)


class AnswerRulesAreConstantTest(unittest.TestCase):
    def test_identical_in_every_episode_and_free_of_episode_data(self):
        episodes = [e for _, e in _generated()] + [load_episode(p) for p in
                                                   sorted((ROOT / "episodes" / "train").iterdir())]
        reference = json.dumps(RULES, sort_keys=True, ensure_ascii=False)
        for episode in episodes:
            rules = build_task(episode["public"]["policy"])["answer_rules"]
            text = json.dumps(rules, sort_keys=True, ensure_ascii=False)
            self.assertEqual(text, reference)
            public = episode["public"]
            ids = {r["row_id"] for r in public["ledger"]} | {d["doc_id"] for d in public["invoices"]}
            ids |= {d["doc_id"] for d in public["payments"]} | {d["invoice_id"] for d in public["invoices"]}
            ids |= {d["payment_id"] for d in public["payments"]} | {d["entity_id"] for d in public["invoices"]}
            ids |= {public["manifest"]["episode_id"]}
            self.assertEqual([i for i in ids if i in text], [])
            for issue in episode["private"]["truth"]["issues"]:
                amount = str(issue["expected_amount_minor"])
                if len(amount) >= 4:
                    self.assertNotIn(amount, text)

    def test_the_rules_have_every_type_and_refer_to_the_policy_for_the_tolerance(self):
        self.assertEqual(list(RULES["types"]), list(ERROR_TYPES))
        for name, rule in RULES["types"].items():
            self.assertIn("policy.amount_tolerance_minor", rule["claimed_amount_minor"], name)
            self.assertTrue(rule["affected_ids"] and rule["currency"], name)
        roles = {token[:-1].split("(")[0] for rule in RULES["types"].values()
                 for tokens in rule["evidence"]["one_of"] for token in tokens}
        self.assertEqual(roles, set(RULES["evidence_roles"]))

    def test_general_rules_follow_the_official_definitions(self):
        text = " ".join(RULES["general"])
        self.assertIn("множества точных ID", text)  # affected_ids and evidence_ids are sets (M01)
        self.assertIn("повтор уже засчитанного вывода — отдельный FP", text)
        self.assertIn("до удаления повторов", text)  # the limit counts the raw list (M05)
        self.assertIn("Несуществующий или недоступный ID", text)  # M04
        self.assertNotIn("evidence_extra_max", text)
        self.assertNotIn("соседние курсы", text)

    def test_every_evidence_set_fits_the_limit_and_the_caller_cannot_alter_the_constants(self):
        from auditgym.answer_rules import answer_rules
        from auditgym.limits import MAX_EVIDENCE_IDS

        for rule in RULES["types"].values():
            for tokens in rule["evidence"]["one_of"]:
                self.assertLessEqual(len(set(tokens)), MAX_EVIDENCE_IDS)
        first = answer_rules()
        first["types"]["fx_error"]["evidence"]["one_of"].clear()
        self.assertTrue(answer_rules()["types"]["fx_error"]["evidence"]["one_of"])


if __name__ == "__main__":
    unittest.main()
