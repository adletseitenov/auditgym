"""Eleventh blind jury (DECISIONS Z): the default journal of make_env() when the working directory is not
writable. The invoice_id/payment_id evidence alias of Z was withdrawn in DECISIONS AA: such an id is exact-compared
and is not an evidence id (FP)."""
from __future__ import annotations

import copy
import os
import pathlib
import stat
import tempfile
import unittest
from unittest import mock

import judge_adapter
from tests.adapter import helpers
from tests.adapter.helpers import closed, new_env, submit_action


def tearDownModule():
    helpers.cleanup()


class DefaultJournalFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers.fixture("C")  # made before tempfile.tempdir is patched: its scratch directory must outlive it

    def run_episode(self, expected_dir: pathlib.Path) -> pathlib.Path:
        fx = helpers.fixture("C")
        env = judge_adapter.make_env()
        expected = expected_dir / "auditgym" / "results.jsonl"
        self.assertEqual(pathlib.Path(env._journal), expected)
        with env:
            observation, _ = env.reset(0, closed(fx["dir"]))
            self.assertIn("task", observation)
            result, _, terminated, _, _ = env.step(submit_action(helpers.empty_findings(fx)))
            self.assertEqual(result["result"], {"accepted": True})
            self.assertTrue(terminated)
            self.assertIsNotNone(env.last_result())
        rows = helpers.finished(expected)
        self.assertEqual(len(rows), 1)
        return expected

    def test_unusable_default_directory_falls_back_to_the_temporary_directory(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as tmp:
            blocker = pathlib.Path(work) / "runs"
            blocker.write_text("not a directory", encoding="utf-8")  # <cwd>/runs cannot be created
            previous = os.getcwd()
            os.chdir(work)
            try:
                with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(tempfile, "tempdir", tmp):
                    os.environ.pop("AUDITGYM_RESULTS", None)
                    self.run_episode(pathlib.Path(tmp))
            finally:
                os.chdir(previous)

    @unittest.skipIf(not hasattr(os, "geteuid") or os.geteuid() == 0, "a root user can write anywhere")
    def test_read_only_working_directory(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as tmp:
            os.chmod(work, stat.S_IRUSR | stat.S_IXUSR)
            previous = os.getcwd()
            os.chdir(work)
            try:
                with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(tempfile, "tempdir", tmp):
                    os.environ.pop("AUDITGYM_RESULTS", None)
                    self.run_episode(pathlib.Path(tmp))
            finally:
                os.chdir(previous)
                os.chmod(work, stat.S_IRWXU)

    def test_a_writable_default_stays_in_the_working_directory(self):
        with tempfile.TemporaryDirectory() as work:
            previous = os.getcwd()
            os.chdir(work)
            try:
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("AUDITGYM_RESULTS", None)
                    env = judge_adapter.make_env()
                    self.assertEqual(os.path.realpath(env._journal),
                                     os.path.realpath(os.path.join(work, "runs", "results.jsonl")))
            finally:
                os.chdir(previous)

    def test_the_results_variable_is_honoured(self):
        with tempfile.TemporaryDirectory() as work:
            target = os.path.join(work, "custom", "r.jsonl")
            with mock.patch.dict(os.environ, {"AUDITGYM_RESULTS": target}):
                self.assertEqual(judge_adapter.make_env()._journal, target)


class DocumentIdIsExactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from auditgym.storage import load_episode

        cls._directory = tempfile.TemporaryDirectory()  # own copy: no cache shared with other test modules
        path = pathlib.Path(cls._directory.name) / "ep"
        helpers.generate_quietly(94111, {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1,
                                                    "period_cutoff": 1}}, str(path))
        cls.episode = {"dir": str(path), "episode": load_episode(path)}

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def setUp(self):
        self.fx = self.episode
        self.public = self.fx["episode"]["public"]
        self.answer = helpers.oracle_findings(self.fx)
        self.assertTrue(self.answer["findings"])

    def aliased(self, field: str) -> dict:
        """The oracle answer with every doc_id of `field` replaced by the invoice_id/payment_id."""
        alias = {d["doc_id"]: d["invoice_id"] for d in self.public["invoices"]}
        alias.update({d["doc_id"]: d["payment_id"] for d in self.public["payments"]})
        message = copy.deepcopy(self.answer)
        changed = 0
        for finding in message["findings"]:
            finding[field] = [alias.get(i, i) for i in finding[field]]
            changed += sum(1 for i in finding[field] if i in alias.values())
        self.assertGreater(changed, 0)
        return message

    def counts(self, message: dict):
        env, path = new_env("alias")
        with env:
            env.reset(0, closed(self.fx["dir"]))
            env.step(submit_action(message))
            row = env.last_result()
        score = judge_adapter.score_episode(self.fx["dir"], message, 5,
                                            journal_path=str(helpers.scratch() / "alias-log.jsonl"))
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (score["tp"], score["fp"], score["fn"]))
        return score["tp"], score["fp"], score["fn"]

    def test_evidence_with_invoice_or_payment_id_is_a_false_positive(self):
        """DECISIONS AA: ids are compared exactly; an invoice_id in evidence_ids names no evidence record."""
        total = len(self.answer["findings"])
        self.assertEqual(self.counts(self.answer), (total, 0, 0))
        tp, fp, fn = self.counts(self.aliased("evidence_ids"))
        self.assertEqual((tp, fp, fn), (0, total, total))

    def test_affected_ids_with_invoice_id_stay_exact_and_are_false_positives(self):
        total = len(self.answer["findings"])
        tp, fp, fn = self.counts(self.aliased("affected_ids"))
        self.assertEqual(tp, 0)
        self.assertEqual((fp, fn), (total, total))


if __name__ == "__main__":
    unittest.main()
