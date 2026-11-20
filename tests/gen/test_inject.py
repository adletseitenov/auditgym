import copy
import datetime as dt
import json
import unittest
from collections import Counter

from auditgym.books import build_books, payment_parts
from auditgym.config import normalize_config
from auditgym.generate import generate
from auditgym.inject import interaction_i1
from auditgym.judge import judge
from auditgym.money import in_period, local_date, parse_time
from auditgym.rng import Rng
from auditgym.storage import episode_files
from auditgym.verify import invoice_signature, predicate_issues, verify
from auditgym.world import build_world, invoice_signature as world_signature

SMALL = {"invoices": [40, 56], "background_pairs": [10, 40]}
ALL = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")


def config(dup=0, gap=0, fx=0, cut=0, **extra):
    errors = {"duplicate_invoice": dup, "allocation_gap": gap, "fx_error": fx, "period_cutoff": cut}
    return {"errors": errors, "size": extra.pop("size", SMALL), **extra}


GRID = [
    ("none", config()),
    ("dup1", config(dup=1, currencies=[])),
    ("cut1", config(cut=1, currencies=[])),
    ("gap1", config(gap=1, currencies=[])),
    ("fx1", config(fx=1, currencies=["USD"])),
    ("dup3_usd", config(dup=3, currencies=["USD"])),
    ("cut3_eur_cny", config(cut=3, currencies=["EUR", "CNY"])),
    ("gap2_fx2_m11", config(gap=2, fx=2, currencies=["USD"], timezone="-11:00")),
    ("all1_p14", config(dup=1, gap=1, fx=1, cut=1, currencies=["USD", "RUB"], timezone="+14:00",
                        period={"start": "2026-12-01", "months": 2})),
    ("all_usd_base", config(dup=1, gap=1, fx=2, cut=1, base_currency="USD", currencies=["KZT", "EUR"],
                            timezone="+05:30")),
    ("i1", config(currencies=["EUR"], interaction="I1", period={"start": "2027-02-01", "months": 1})),
    ("i2", config(cut=1, interaction="I2", timezone="+05:45")),
    ("no_distractors", config(dup=1, gap=1, fx=1, cut=1, distractors=False)),
]
MAX_CUTOFF = config(cut=5, size={"invoices": [40, 60], "background_pairs": [0, 20]})


def by_id(records, key):
    return {record[key]: record for record in records}


def issues_of(episode, kind=None):
    return [i for i in episode["private"]["truth"]["issues"] if kind is None or i["type"] == kind]


def posted_payments(public):
    """invoice_id -> number of posted payments referring to it."""
    posted = {r["payment_id"] for r in public["ledger"] if r["payment_id"]}
    out = Counter()
    for payment in public["payments"]:
        if payment["payment_id"] in posted and payment["amount_minor"] > 0:
            for ref in payment["invoice_refs"] or []:
                out[ref] += 1
    return out


class GridTest(unittest.TestCase):
    """verify() is empty and the truth has exactly the configured issues."""

    def test_grid(self):
        for name, cfg in GRID:
            for seed in range(25):
                with self.subTest(config=name, seed=seed):
                    episode = generate(seed, cfg)
                    self.assertEqual(verify(episode), [])
                    counts = Counter(issue["type"] for issue in issues_of(episode))
                    expected = Counter({k: v for k, v in cfg["errors"].items() if v})
                    expected.update({"I1": ["allocation_gap", "fx_error"], "I2": ["duplicate_invoice", "period_cutoff"]}
                                    .get(cfg.get("interaction"), []))
                    self.assertEqual(counts, expected)
                    found, undefined = predicate_issues(episode["public"])
                    self.assertEqual(undefined, [])
                    self.assertEqual(len(found), len(issues_of(episode)))
                    self.assert_ledger_diff(episode)

    def assert_ledger_diff(self, episode):
        """pre_ledger and ledger differ exactly by the rows the steps record."""
        pre = by_id(episode["private"]["pre_ledger"], "row_id")
        post = by_id(episode["public"]["ledger"], "row_id")
        touched = set()
        for step in episode["private"]["provenance"]["steps"]:
            touched |= {r["row_id"] for r in step["before"] + step["after"] if "row_id" in r}
        changed = {rid for rid in set(pre) | set(post) if pre.get(rid) != post.get(rid)}
        self.assertLessEqual(changed, touched)


class OperatorTest(unittest.TestCase):
    def test_duplicate_is_a_copy_with_new_ids_and_row_evidence(self):
        for seed in range(30):
            episode = generate(seed, config(dup=3, currencies=["USD"]))
            invoices = by_id(episode["public"]["invoices"], "doc_id")
            rows = episode["public"]["ledger"]
            originals = set()
            for issue in issues_of(episode):
                a, b = (invoices[doc] for doc in issue["affected_ids"])
                self.assertNotEqual(a["invoice_id"], b["invoice_id"])
                self.assertNotEqual(a["issued_at"], b["issued_at"])
                self.assertEqual(invoice_signature(a), invoice_signature(b))
                self.assertEqual(issue["expected_amount_minor"], a["total_minor"])
                own_a = {r["row_id"] for r in rows if r["invoice_id"] == a["invoice_id"] and not r["payment_id"]}
                own_b = {r["row_id"] for r in rows if r["invoice_id"] == b["invoice_id"] and not r["payment_id"]}
                self.assertEqual(len(issue["evidence_sets"]), 4)
                self.assertEqual({frozenset(s) for s in issue["evidence_sets"]},
                                 {frozenset([a["doc_id"], b["doc_id"], x, y]) for x in own_a for y in own_b})
                self.assertTrue(originals.isdisjoint(issue["affected_ids"]))  # one copy per original
                originals |= set(issue["affected_ids"])

    def test_cutoff_posts_an_unposted_neighbouring_period_invoice(self):
        """D4: the target is a legitimately unposted invoice of the neighbouring period; its document
        does not change and the ledger gains exactly its two rows at the policy amount."""
        directions, near, on_edge, paid = Counter(), 0, 0, 0
        for seed in range(120):
            episode = generate(seed, config(cut=1, currencies=["USD"], timezone="-11:00"))
            public, private = episode["public"], episode["private"]
            policy = public["policy"]
            step = private["provenance"]["steps"][0]
            params = step["params"]
            directions[params["direction"]] += 1
            paid += params["paid"]
            (issue,) = issues_of(episode)
            doc = issue["affected_ids"][0]
            invoice = by_id(public["invoices"], "doc_id")[doc]
            self.assertEqual(by_id(private["pre_invoices"], "doc_id")[doc], invoice)  # document unchanged
            self.assertEqual([r for r in step["before"] if "line_items" in r], [r for r in step["after"] if "line_items" in r])
            self.assertFalse(in_period(invoice["issued_at"], policy["period_start"], policy["period_end"]))
            edge = policy["period_start"] if params["direction"] == "before_start" else policy["period_end"]
            distance = abs(parse_time(invoice["issued_at"]) - parse_time(edge))
            self.assertLess(distance, dt.timedelta(days=4))  # ruling G3
            near += distance <= dt.timedelta(hours=6)
            on_edge += distance <= dt.timedelta(seconds=1)
            first, last = local_date(policy["period_start"], "-11:00"), "2026-10-31"
            for item in invoice["line_items"]:
                self.assertTrue(item["service_end"] < first or item["service_start"] > last)
            added = [r for r in step["after"] if "row_id" in r]
            self.assertEqual([r for r in step["before"] if "row_id" in r], [])
            self.assertEqual(sorted(r["account"] for r in added), ["receivable", "revenue"])
            pre_ids = {r["row_id"] for r in private["pre_ledger"]}
            self.assertEqual({r["row_id"] for r in public["ledger"]} - pre_ids, {r["row_id"] for r in added})
            rate = by_id(public["fx"], "date").get(local_date(invoice["issued_at"], policy["timezone"]))
            if invoice["currency"] == policy["base_currency"]:
                self.assertEqual({r["debit_minor"] + r["credit_minor"] for r in added}, {invoice["total_minor"]})
            else:
                self.assertIsNotNone(rate)
            self.assertEqual(sorted(map(sorted, issue["evidence_sets"])),
                             sorted(sorted([doc, r["row_id"]]) for r in added))
        self.assertGreater(directions["before_start"], 35)
        self.assertGreater(directions["after_end"], 35)
        self.assertGreater(near, 30)
        self.assertGreater(on_edge, 3)
        self.assertGreater(paid, 40)  # paid targets are common

    def test_fx_error_modes_and_threshold(self):
        modes, payments, invoices = Counter(), 0, 0
        for seed in range(80):
            episode = generate(seed, config(fx=2, currencies=["USD", "EUR"]))
            docs = by_id(episode["public"]["invoices"] + episode["public"]["payments"], "doc_id")
            for step, issue in zip(episode["private"]["provenance"]["steps"], issues_of(episode)):
                modes[step["params"]["mode"]] += 1
                doc = docs[issue["affected_ids"][0]]
                self.assertGreaterEqual(issue["expected_amount_minor"], 100)
                if "line_items" in doc:
                    invoices += 1
                else:
                    payments += 1
                    self.assertGreater(doc["amount_minor"], 0)  # ruling 5: no refunds or advances as targets
                    self.assertIsNotNone(doc["invoice_refs"])
                    self.assertEqual(len(issue["evidence_sets"]), 1)
                key = f"{local_date(doc.get('issued_at', doc.get('timestamp')), 'policy' and episode['public']['policy']['timezone'])}" \
                      f"|{doc['currency']}|{episode['public']['policy']['base_currency']}"
                for group in issue["evidence_sets"]:  # D1: doc, one conversion row, the fx key
                    self.assertEqual(len(group), 3)
                    self.assertIn(doc["doc_id"], group)
                    self.assertIn(key, group)
                self.assertEqual(len(issue["evidence_sets"]), 1 if "payment_id" in doc else 2)
        self.assertGreater(modes["wrong_date"], 20)
        self.assertGreater(modes["wrong_rate"], 20)
        self.assertGreater(payments, 10)
        self.assertGreater(invoices, 10)

    def test_allocation_gap_shape(self):
        for seed in range(60):
            episode = generate(seed, config(gap=2, cut=2, currencies=["USD"]))
            public = episode["public"]
            docs = by_id(public["invoices"] + public["payments"], "doc_id")
            cut_docs = {d for i in issues_of(episode, "period_cutoff") for d in i["affected_ids"]}
            for issue in issues_of(episode, "allocation_gap"):
                self.assertFalse(cut_docs & set(issue["affected_ids"]))  # ruling 4
                payment = next(docs[d] for d in issue["affected_ids"] if "payment_id" in docs[d])
                invoice = next(docs[d] for d in issue["affected_ids"] if "line_items" in docs[d])
                self.assertEqual(payment["invoice_refs"], [invoice["invoice_id"]])
                self.assertTrue(in_period(invoice["issued_at"], public["policy"]["period_start"],
                                          public["policy"]["period_end"]))
                unallocated = [r for r in public["ledger"] if r["payment_id"] == payment["payment_id"]
                               and r["account"] == "unallocated"]
                self.assertEqual(len(unallocated), 1)
                self.assertEqual(unallocated[0]["credit_minor"], issue["expected_amount_minor"])
                debit = [r["row_id"] for r in public["ledger"] if r["invoice_id"] == invoice["invoice_id"]
                         and not r["payment_id"] and r["account"] == "receivable"]
                credit = [r["row_id"] for r in public["ledger"] if r["payment_id"] == payment["payment_id"]
                          and r["account"] == "receivable"]
                self.assertEqual(issue["evidence_sets"],  # D1
                                 [sorted([payment["doc_id"], invoice["doc_id"]]) + sorted(
                                     [unallocated[0]["row_id"], debit[0], credit[0]])])
                others = [p for p in public["payments"] if invoice["invoice_id"] in (p["invoice_refs"] or [])]
                self.assertEqual(others, [payment])

    def test_one_error_per_document(self):
        for seed in range(60):
            episode = generate(seed, config(dup=1, gap=1, fx=2, cut=1, currencies=["USD"]))
            docs = Counter(d for i in issues_of(episode) for d in i["affected_ids"])
            self.assertEqual(max(docs.values()), 1)

    def test_operators_have_their_own_streams(self):
        """On one story, adding another type moves a type's targets only when they collide."""
        from auditgym.inject import inject_errors
        same = 0
        for seed in range(40):
            cfg = normalize_config(config(cut=2, currencies=["USD"]))
            world = build_world(Rng(seed, "world"), cfg)
            alone = inject_errors(Rng(seed), world, cfg)[1]
            both = inject_errors(Rng(seed), world, dict(cfg, errors=dict(cfg["errors"], fx_error=2)))[1]
            keys = [sorted(i["affected"] for i in injections if i["type"] == "period_cutoff")
                    for injections in (alone, both)]
            same += keys[0] == keys[1]
        self.assertGreater(same, 36)

    def test_retries_are_deterministic_and_recorded(self):
        tight = {"errors": {"fx_error": 1}, "currencies": ["EUR"], "size": {"invoices": [10, 10], "background_pairs": [0, 5]}}
        attempts = Counter()
        for seed in range(150):
            episode = generate(seed, tight)
            provenance = episode["private"]["provenance"]
            attempts[provenance["attempt"]] += 1
            self.assertEqual(len(provenance["retries"]), provenance["attempt"])
            self.assertTrue(all(not r["reason"].startswith("internal") for r in provenance["retries"]))
        self.assertGreater(sum(n for a, n in attempts.items() if a > 0), 0)
        seed = next(s for s in range(150) if generate(s, tight)["private"]["provenance"]["attempt"] > 0)
        self.assertEqual(episode_files(generate(seed, tight)), episode_files(generate(seed, tight)))


class RulingTest(unittest.TestCase):
    def test_recurring_twins_split_the_period_service(self):
        """D3: a recurring twin bills the other half of the period's service, never a neighbouring month."""
        twins = 0
        for seed in range(30):
            public = generate(seed, {"errors": {}, "currencies": ["USD"]})["public"]
            policy = public["policy"]
            self.assertEqual(policy["duplicate_signature"], "entity, currency, total, line items incl. service period; service_start/service_end only identify duplicates and never affect period_cutoff, which uses the document date alone (period_basis)")
            first, last = "2026-10-01", "2026-10-31"
            posted = {r["invoice_id"] for r in public["ledger"] if r["invoice_id"] and not r["payment_id"]}
            loose = Counter()
            for invoice in public["invoices"]:
                if invoice["invoice_id"] not in posted:
                    continue
                for item in invoice["line_items"]:
                    self.assertTrue(first <= item["service_start"] <= item["service_end"] <= last, item)
                loose[(invoice["entity_id"], invoice["total_minor"],
                       json.dumps(sorted((x["description"], x["quantity"]) for x in invoice["line_items"])))] += 1
            twins += sum(1 for n in loose.values() if n == 2)
            self.assertEqual(predicate_issues(public), ([], []))
        self.assertGreater(twins, 20)

    def test_i2_copy_keeps_line_items(self):
        """Ruling 2."""
        for seed in range(30):
            episode = generate(seed, config(interaction="I2"))
            dup = issues_of(episode, "duplicate_invoice")[0]
            cut = issues_of(episode, "period_cutoff")[0]
            invoices = by_id(episode["public"]["invoices"], "doc_id")
            self.assertIn(cut["affected_ids"][0], dup["affected_ids"])
            a, b = (invoices[d] for d in dup["affected_ids"])
            self.assertEqual(a["line_items"], b["line_items"])

    def test_policy_values_of_advances_and_refunds(self):
        """Ruling 5: advances and refunds are posted at the policy value, so fx_error does not fire."""
        refunds = advances = 0
        for seed in range(30):
            public = generate(seed, {"errors": {}, "currencies": ["USD", "EUR"]})["public"]
            refunds += sum(p["amount_minor"] < 0 for p in public["payments"])
            advances += sum(p["invoice_refs"] is None for p in public["payments"])
            self.assertEqual(predicate_issues(public), ([], []))
        self.assertGreater(refunds, 3)
        self.assertGreater(advances, 3)


class InteractionTest(unittest.TestCase):
    def test_i1(self):
        boundary = 0
        for seed in range(40):
            episode = generate(seed, config(currencies=["USD"], interaction="I1"))
            gap = issues_of(episode, "allocation_gap")[0]
            fx = issues_of(episode, "fx_error")[0]
            public = episode["public"]
            payment = by_id(public["payments"], "doc_id")[fx["affected_ids"][0]]
            invoice = next(i for i in public["invoices"] if i["invoice_id"] == payment["invoice_refs"][0])
            self.assertEqual(sorted([payment["doc_id"], invoice["doc_id"]]), gap["affected_ids"])
            self.assertLess(payment["amount_minor"], invoice["total_minor"])  # partial
            self.assertNotEqual(payment["currency"], public["policy"]["base_currency"])
            end = parse_time(public["policy"]["period_end"])
            boundary += end - parse_time(payment["timestamp"]) <= dt.timedelta(hours=48)
            self.assertEqual(len(gap["evidence_sets"][0]), 5)  # D1
        self.assertEqual(boundary, 40)

    def test_i1_payment_offset_is_drawn_like_any_payment(self):
        """D8: the I1 payment is written in another offset as often as an ordinary payment."""
        i1 = i1_other = ordinary = ordinary_other = 0
        for seed in range(200):
            episode = generate(seed, config(currencies=["USD"], interaction="I1"))
            public = episode["public"]
            fx = issues_of(episode, "fx_error")[0]["affected_ids"][0]
            zone = public["policy"]["timezone"]
            for payment in public["payments"]:
                other = not payment["timestamp"].endswith(zone)
                if payment["doc_id"] == fx:
                    i1, i1_other = i1 + 1, i1_other + other
                else:
                    ordinary, ordinary_other = ordinary + 1, ordinary_other + other
        self.assertLess(abs(i1_other / i1 - ordinary_other / ordinary), 0.07, (i1_other / i1, ordinary_other / ordinary))

    def test_i1_amounts_do_not_depend_on_order(self):
        for seed in range(20):
            cfg = normalize_config(config(currencies=["USD"], interaction="I1"))
            world = build_world(Rng(seed, "story"), cfg)
            order = sorted(inv["key"] for inv in world["invoices"])
            (first, gap_issue), (second, fx_issue) = interaction_i1(Rng(seed, "i1"), world, order, set())
            pkey = fx_issue["affected"][0][2:]
            payment = next(p for p in second["payments"] if p["key"] == pkey)
            fx_only = copy.deepcopy(second)
            next(p for p in fx_only["payments"] if p["key"] == pkey)["gap_minor"] = 0
            part = payment_parts(fx_only)[pkey]
            self.assertEqual(abs(part["cash"] - part["correct_cash"]), fx_issue["amount_minor"])
            rows = {r["key"]: r for r in build_books(second)}
            self.assertEqual(rows[f"R:{pkey}:un"]["credit_minor"], gap_issue["amount_minor"])
            gap_first = {r["key"]: r for r in build_books(first)}[f"R:{pkey}:un"]["credit_minor"]
            self.assertEqual(gap_first, gap_issue["amount_minor"])
            self.assertIsNotNone(payment["rate_override"])

    def test_i2(self):
        for seed in range(30):
            episode = generate(seed, config(interaction="I2"))
            public = episode["public"]
            cut = issues_of(episode, "period_cutoff")[0]
            copy_invoice = by_id(public["invoices"], "doc_id")[cut["affected_ids"][0]]
            self.assertFalse(in_period(copy_invoice["issued_at"], public["policy"]["period_start"],
                                       public["policy"]["period_end"]))
            rows = [r for r in public["ledger"] if r["invoice_id"] == copy_invoice["invoice_id"]]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(issues_of(episode)), 2)
            for step in episode["private"]["provenance"]["steps"]:  # D7: the scenario is not an id
                self.assertEqual(step["params"]["scenario"], "I2")

    def test_i2_copy_is_paid_like_any_invoice_outside_the_period(self):
        """D8: the copy is paid as often as a legitimate invoice of the neighbouring period."""
        copies = paid_copies = outside = paid_outside = 0
        for seed in range(200):
            episode = generate(seed, config(interaction="I2"))
            public, policy = episode["public"], episode["public"]["policy"]
            copy_id = issues_of(episode, "period_cutoff")[0]["affected_ids"][0]
            refs = Counter(ref for p in public["payments"] for ref in p["invoice_refs"] or [])
            for invoice in public["invoices"]:
                if in_period(invoice["issued_at"], policy["period_start"], policy["period_end"]):
                    continue
                if invoice["doc_id"] == copy_id:
                    copies, paid_copies = copies + 1, paid_copies + bool(refs[invoice["invoice_id"]])
                else:
                    outside, paid_outside = outside + 1, paid_outside + bool(refs[invoice["invoice_id"]])
        self.assertGreater(paid_outside, 50)  # L6: a paid invoice outside the period is often legitimate
        self.assertLess(abs(paid_copies / copies - paid_outside / outside), 0.1, (paid_copies / copies,
                                                                                 paid_outside / outside))


class DistractorTest(unittest.TestCase):
    def kinds(self, public):
        policy = public["policy"]
        inside = lambda t: in_period(t, policy["period_start"], policy["period_end"])  # noqa: E731
        invoices = by_id(public["invoices"], "invoice_id")
        posted = {r["payment_id"] for r in public["ledger"] if r["payment_id"]}
        posted_invoices = {r["invoice_id"] for r in public["ledger"] if r["invoice_id"] and not r["payment_id"]}
        out = Counter()
        refs = Counter(ref for p in public["payments"] if p["payment_id"] in posted and p["amount_minor"] > 0
                       for ref in p["invoice_refs"] or [])
        for payment in public["payments"]:
            single = payment["invoice_refs"] and len(payment["invoice_refs"]) == 1
            total = invoices[payment["invoice_refs"][0]]["total_minor"] if single else None
            out["advance"] += payment["invoice_refs"] is None
            out["refund"] += payment["amount_minor"] < 0
            out["overpayment"] += bool(single and payment["amount_minor"] > total)
            out["partial"] += bool(single and 0 < payment["amount_minor"] < total)
            out["unposted_payment"] += payment["payment_id"] not in posted
        out["split"] = sum(1 for n in refs.values() if n == 2)
        out["outside_invoice"] = sum(1 for i in public["invoices"] if not inside(i["issued_at"]))
        out["other_offset"] = sum(1 for i in public["invoices"] if not i["issued_at"].endswith(policy["timezone"]))
        loose = Counter((i["entity_id"], i["currency"], i["total_minor"],
                         json.dumps(sorted((x["description"], x["quantity"], x["unit_price_minor"]) for x in i["line_items"])))
                        for i in public["invoices"] if i["invoice_id"] in posted_invoices)
        out["recurring"] = sum(1 for n in loose.values() if n > 1)
        return out

    def test_every_distractor_appears_and_stays_clean(self):
        total = Counter()
        for seed in range(30):
            public = generate(seed, {"errors": {}, "currencies": ["USD"]})["public"]
            self.assertEqual(predicate_issues(public), ([], []))
            total.update(self.kinds(public))
        for kind in ("advance", "refund", "overpayment", "partial", "split", "outside_invoice", "other_offset",
                     "recurring", "unposted_payment"):
            self.assertGreater(total[kind], 3, kind)

    def test_without_distractors_there_are_none(self):
        for seed in range(30):
            public = generate(seed, {"errors": {}, "currencies": ["USD"], "distractors": False})["public"]
            kinds = self.kinds(public)
            self.assertEqual({k: v for k, v in kinds.items() if v}, {}, seed)


class LeakTest(unittest.TestCase):
    """Payment status and posting time say nothing about where the errors are."""

    def test_paid_share_of_cutoff_targets_matches_the_story(self):
        targets = paid_targets = population = paid_population = 0
        for seed in range(150):
            episode = generate(seed, MAX_CUTOFF)
            pre = dict(episode["public"], ledger=episode["private"]["pre_ledger"],
                       invoices=episode["private"]["pre_invoices"], payments=episode["private"]["pre_payments"])
            policy = pre["policy"]
            refs = Counter(ref for p in pre["payments"] for ref in p["invoice_refs"] or [])
            for invoice in pre["invoices"]:  # the pool: invoices of the neighbouring periods (D4)
                if not in_period(invoice["issued_at"], policy["period_start"], policy["period_end"]):
                    population += 1
                    paid_population += bool(refs[invoice["invoice_id"]])
            for step in episode["private"]["provenance"]["steps"]:
                targets += 1
                paid_targets += step["params"]["paid"]
        share, base = paid_targets / targets, paid_population / population
        self.assertLess(abs(share - base), 0.06, (share, base))

    def test_unpaid_share_of_duplicate_copies_matches_the_story(self):
        copies = unpaid_copies = population = unpaid_population = 0
        for seed in range(150):
            episode = generate(seed, config(dup=5, size={"invoices": [40, 60], "background_pairs": [0, 20]}))
            public, policy = episode["public"], episode["public"]["policy"]
            pre_ids = {i["invoice_id"] for i in episode["private"]["pre_invoices"]}
            paid = posted_payments(public)
            for invoice in public["invoices"]:
                if not in_period(invoice["issued_at"], policy["period_start"], policy["period_end"]):
                    continue
                if invoice["invoice_id"] in pre_ids:
                    population += 1
                    unpaid_population += invoice["invoice_id"] not in paid
                else:
                    copies += 1
                    unpaid_copies += invoice["invoice_id"] not in paid
        share, base = unpaid_copies / copies, unpaid_population / population
        self.assertLess(abs(share - base), 0.06, (share, base))

    def test_posting_time_differs_from_document_time_says_nothing(self):
        """Rule "posting moment != document moment": every posted document is flagged, so its precision
        is the base rate. Rule "posting date != document date" (company dates): apart from cutoffs, which
        the definition itself finds, its precision is no better than the base rate either."""
        flagged = hits = posted = errors = 0
        date_flagged = date_hits = non_cut_posted = non_cut_errors = 0
        for seed in range(80):
            episode = generate(seed, config(dup=1, gap=1, fx=2, cut=1, currencies=["USD"]))
            public, offset = episode["public"], episode["public"]["policy"]["timezone"]
            affected = {d for i in issues_of(episode) for d in i["affected_ids"]}
            cut = {d for i in issues_of(episode, "period_cutoff") for d in i["affected_ids"]}
            post = {}
            for row in public["ledger"]:
                if row["payment_id"]:
                    post[("p", row["payment_id"])] = row["timestamp"]
                elif row["invoice_id"]:
                    post[("i", row["invoice_id"])] = row["timestamp"]
            docs = [(i["doc_id"], i["issued_at"], post.get(("i", i["invoice_id"]))) for i in public["invoices"]]
            docs += [(p["doc_id"], p["timestamp"], post.get(("p", p["payment_id"]))) for p in public["payments"]]
            for doc, dated, posted_at in docs:
                if posted_at is None:
                    continue
                posted += 1
                errors += doc in affected
                if parse_time(posted_at) != parse_time(dated):
                    flagged += 1
                    hits += doc in affected
                if doc in cut:
                    continue
                non_cut_posted += 1
                non_cut_errors += doc in affected
                if local_date(posted_at, offset) != local_date(dated, offset):
                    date_flagged += 1
                    date_hits += doc in affected
        self.assertEqual(flagged, posted)
        self.assertLessEqual(hits / flagged, errors / posted + 1e-9)
        self.assertLess(date_hits / date_flagged, non_cut_errors / non_cut_posted * 1.5 + 0.01)

    def answer(self, episode, invoices):
        rows = {}
        for row in episode["public"]["ledger"]:
            if row["invoice_id"] and not row["payment_id"] and row["debit_minor"]:
                rows[row["invoice_id"]] = row["row_id"]
        findings = [{"type": "period_cutoff", "affected_ids": [i["doc_id"]], "claimed_amount_minor": i["total_minor"],
                     "currency": i["currency"], "evidence_ids": [i["doc_id"], rows[i["invoice_id"]]],
                     "explanation": ""} for i in invoices if i["invoice_id"] in rows]
        return {"episode_id": episode["public"]["manifest"]["episode_id"], "findings": findings}

    def test_flagging_unpaid_or_all_invoices_as_cutoff_does_not_pay(self):
        """Literal: silence is -1 on books with errors, so any true positive beats it; the meaningful
        reading is that flagging the unpaid invoices is no better than flagging as many random posted
        invoices, and that flagging every invoice never beats silence (> 30 findings) and, capped at
        30, stays negative. Invoices dated outside the period are left out of the unpaid/random
        comparison: the definition itself finds every posted one (D4), with or without payment."""
        unpaid_rewards, random_rewards = [], []
        for seed in range(120):
            for cfg in (MAX_CUTOFF, config(cut=2, dup=1, gap=1, fx=1, currencies=["USD"])):
                episode = generate(seed, cfg)
                public = episode["public"]
                posted = {r["invoice_id"] for r in public["ledger"] if r["invoice_id"] and not r["payment_id"]}
                invoices = [i for i in public["invoices"] if i["invoice_id"] in posted]
                paid = posted_payments(public)
                policy = public["policy"]
                dated_inside = [i for i in invoices if in_period(i["issued_at"], policy["period_start"], policy["period_end"])]
                unpaid = [i for i in dated_inside if i["invoice_id"] not in paid][:30]
                pool = list(dated_inside)
                Rng(seed, "leak").shuffle(pool)
                silence = judge(episode, {"episode_id": public["manifest"]["episode_id"], "findings": []}, 10)["reward"]
                unpaid_rewards.append(judge(episode, self.answer(episode, unpaid), 10)["reward"])
                random_rewards.append(judge(episode, self.answer(episode, pool[:len(unpaid)]), 10)["reward"])
                everything = judge(episode, self.answer(episode, invoices), 10)["reward"]
                self.assertLessEqual(everything, silence)
                everyone = list(invoices)
                Rng(seed, "leak").shuffle(everyone)
                capped = judge(episode, self.answer(episode, everyone[:30]), 10)["reward"]
                self.assertLess(capped, 0)
        mean_unpaid = sum(unpaid_rewards) / len(unpaid_rewards)
        mean_random = sum(random_rewards) / len(random_rewards)
        self.assertLessEqual(mean_unpaid, mean_random + 0.03, (mean_unpaid, mean_random))


class NotSeparableTest(unittest.TestCase):
    """Injected records look like any other record apart from the error itself."""

    def test_surface_features(self):
        ranks, line_counts_injected, line_counts_all = [], Counter(), Counter()
        for seed in range(80):
            episode = generate(seed, config(dup=1, gap=1, fx=1, cut=1, currencies=["USD"]))
            public = episode["public"]
            ids = sorted(episode["private"]["provenance"]["id_map"].values())
            injected = set()
            for step in episode["private"]["provenance"]["steps"]:
                for record in step["after"]:
                    injected |= {record.get("doc_id"), record.get("invoice_id"), record.get("row_id")} - {None}
            for public_id in injected:
                ranks.append(ids.index(public_id) / (len(ids) - 1))
            for invoice in public["invoices"]:
                line_counts_all[len(invoice["line_items"])] += 1
                if invoice["doc_id"] in injected:
                    line_counts_injected[len(invoice["line_items"])] += 1
        mean = sum(ranks) / len(ranks)
        self.assertTrue(0.45 < mean < 0.55, mean)
        self.assertTrue(0.4 < sum(r < 0.5 for r in ranks) / len(ranks) < 0.6)
        self.assertEqual(set(line_counts_injected), set(line_counts_all))

    def test_injected_invoice_position_is_spread(self):
        positions = []
        for seed in range(120):
            episode = generate(seed, config(dup=1, currencies=["USD"]))
            copy_doc = next(r["doc_id"] for r in episode["private"]["provenance"]["steps"][0]["after"]
                            if "line_items" in r)
            docs = [inv["doc_id"] for inv in episode["public"]["invoices"]]
            positions.append(docs.index(copy_doc) / (len(docs) - 1))
        self.assertTrue(0.4 < sum(positions) / len(positions) < 0.6)
        self.assertGreater(sum(p < 0.25 for p in positions), 15)
        self.assertGreater(sum(p > 0.75 for p in positions), 15)


class VerifyCatchesBrokenTruthTest(unittest.TestCase):
    def setUp(self):
        self.episode = generate(5, config(dup=1, gap=1, fx=1, cut=1, currencies=["USD"]))
        self.assertEqual(verify(self.episode), [])

    def broken(self, change):
        episode = copy.deepcopy(self.episode)
        change(episode)
        return verify(episode)

    def issue(self, episode, kind):
        return next(i for i in episode["private"]["truth"]["issues"] if i["type"] == kind)

    def test_wrong_amount(self):
        for kind in ALL:
            def change(e, kind=kind):
                self.issue(e, kind)["expected_amount_minor"] += 1
            self.assertTrue(any("deviation" in p for p in self.broken(change)), kind)

    def test_missing_evidence_id(self):
        for kind in ALL:
            def change(e, kind=kind):
                self.issue(e, kind)["evidence_sets"][0][-1] = "nosuchid00"
            self.assertTrue(self.broken(change), kind)

    def test_too_much_evidence(self):
        def change(e):
            rows = [r["row_id"] for r in e["public"]["ledger"][:10]]
            self.issue(e, "duplicate_invoice")["evidence_sets"][0] += rows
        self.assertTrue(any("evidence" in p for p in self.broken(change)))

    def test_document_only_duplicate_evidence(self):
        def change(e):
            issue = self.issue(e, "duplicate_invoice")
            issue["evidence_sets"] = [issue["affected_ids"]]
        self.assertTrue(any("evidence" in p for p in self.broken(change)))

    def test_incomplete_cutoff_evidence(self):
        def change(e):
            issue = self.issue(e, "period_cutoff")
            issue["evidence_sets"] = issue["evidence_sets"][:1]
        self.assertTrue(self.broken(change))

    def test_unlisted_issue(self):
        for kind in ALL:
            def change(e, kind=kind):
                truth = e["private"]["truth"]
                gone = self.issue(e, kind)["issue_id"]
                truth["issues"] = [i for i in truth["issues"] if i["issue_id"] != gone]
                e["private"]["provenance"]["steps"] = [s for s in e["private"]["provenance"]["steps"]
                                                       if s["issue_id"] != gone]
            problems = self.broken(change)  # a posting issue removed from the truth also breaks the clean posting rules
            self.assertTrue(any("no issue lists it" in p or "payment rule" in p or "policy amount" in p
                                or "outside the period but posted" in p for p in problems), (kind, problems[:2]))

    def test_pre_snapshot_tampered(self):
        for table in ("pre_ledger", "pre_invoices", "pre_payments"):
            def change(e, table=table):
                e["private"][table].pop()
            self.assertTrue(self.broken(change), table)

    def test_affected_ids_not_a_document(self):
        def change(e):
            self.issue(e, "duplicate_invoice")["affected_ids"][0] = e["public"]["ledger"][0]["row_id"]
        self.assertTrue(self.broken(change))


class FixtureTest(unittest.TestCase):
    def test_hand_written_fixture_passes_verify(self):
        import pathlib
        from auditgym.storage import load_episode
        root = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "tiny"
        self.assertEqual(verify(load_episode(root)), [])


class ReproducibilityTest(unittest.TestCase):
    CFG = config(dup=1, gap=1, fx=1, cut=1, currencies=["USD", "EUR"], timezone="+14:00")

    def test_same_seed_same_bytes(self):
        for seed in range(10):
            self.assertEqual(episode_files(generate(seed, self.CFG)), episode_files(generate(seed, self.CFG)))

    def canonical(self, episode):
        back = {public: internal for internal, public in episode["private"]["provenance"]["id_map"].items()}

        def rename(value):
            if isinstance(value, dict):
                return {key: rename(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rename(item) for item in value]
            return back.get(value, value) if isinstance(value, str) else value

        out = {}
        for name in ("ledger", "invoices", "payments"):
            records = rename(copy.deepcopy(episode["public"][name]))
            for record in records:
                if record.get("invoice_refs"):
                    record["invoice_refs"].sort()
            out[name] = sorted(json.dumps(record, sort_keys=True) for record in records)
        for name in ("pre_ledger", "pre_invoices", "pre_payments"):
            records = rename(copy.deepcopy(episode["private"][name]))
            for record in records:
                if record.get("invoice_refs"):
                    record["invoice_refs"].sort()
            out[name] = sorted(json.dumps(r, sort_keys=True) for r in records)
        issues = rename(copy.deepcopy(episode["private"]["truth"]["issues"]))
        for issue in issues:
            issue["affected_ids"].sort()
            issue["evidence_sets"] = sorted(sorted(group) for group in issue["evidence_sets"])
        out["issues"] = issues
        steps = rename(copy.deepcopy(episode["private"]["provenance"]["steps"]))
        for step in steps:
            step["before"] = sorted(json.dumps(r, sort_keys=True) for r in step["before"])
            step["after"] = sorted(json.dumps(r, sort_keys=True) for r in step["after"])
        out["steps"] = json.dumps(steps, sort_keys=True)
        out["fx"] = episode["public"]["fx"]
        return out

    def test_surface_seed_renames_episode_and_truth(self):
        for seed in range(10):
            first = generate(seed, dict(self.CFG, surface_seed=1))
            second = generate(seed, dict(self.CFG, surface_seed=2))
            self.assertEqual(verify(second), [])
            self.assertNotEqual(first["private"]["truth"]["issues"][0]["affected_ids"],
                                second["private"]["truth"]["issues"][0]["affected_ids"])
            self.assertEqual(self.canonical(first), self.canonical(second))


if __name__ == "__main__":
    unittest.main()
