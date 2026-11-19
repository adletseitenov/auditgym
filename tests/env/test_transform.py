"""CASE §06/§10: «перестановка и смена ID сохраняет ответ oracle и награду с учётом новых ID».
tools/transform.py is the external reorder-and-rename; tools/metamorph.py checks the pairs."""
import ast
import io
import json
import pathlib
import shutil
import tempfile
import unittest

from auditgym.evaluator import Evaluator
from auditgym.generate import generate
from auditgym.storage import load_episode, write_episode
from tests.env.helpers import ROOT, TINY, FakeClock
from tools import metamorph
from tools.transform import ID_STYLES, TransformError, load_mapping, transform

CONFIG = {"errors": {"duplicate_invoice": 2, "period_cutoff": 2}, "currencies": ["USD", "EUR"]}


def _reward(directory, answer) -> float:
    evaluator = Evaluator(clock=FakeClock())
    evaluator.load(directory)
    return evaluator.step({"action": "submit", "findings": answer})[1]


def _tree(directory) -> dict:
    root = pathlib.Path(directory)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


class TransformTest(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="transform_test_"))
        self.episode = generate(424242, CONFIG)
        self.src = self.tmp / "src"
        write_episode(self.episode, self.src)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_seed_same_bytes_other_seed_other_bytes(self):
        transform(self.src, self.tmp / "a", 9)
        transform(self.src, self.tmp / "b", 9)
        transform(self.src, self.tmp / "c", 10)
        self.assertEqual(_tree(self.tmp / "a"), _tree(self.tmp / "b"))
        self.assertNotEqual(_tree(self.tmp / "a"), _tree(self.tmp / "c"))

    def test_every_old_id_is_gone_and_the_oracle_reward_is_kept(self):
        oracle = metamorph.oracle_answer(self.episode)
        for number, style in enumerate(ID_STYLES):
            dst = self.tmp / style
            document = transform(self.src, dst, number, id_style=style, unsorted_truth=bool(number % 2))
            mapping = load_mapping(dst)
            self.assertEqual(len(set(mapping.values())), len(mapping), "mapping must be a bijection")
            text = b"".join(content for path, content in _tree(dst).items() if path != "mapping.json").decode()
            for old in document["ids"]:
                self.assertNotIn(f'"{old}"', text)
                self.assertNotIn(f",{old},", text)
            moved = load_episode(dst)
            self.assertNotEqual(moved["public"]["ledger"], self.episode["public"]["ledger"])
            self.assertEqual(_reward(dst, metamorph.map_answer(oracle, mapping)), _reward(self.src, oracle))
            self.assertEqual(_reward(dst, metamorph.map_answer(oracle, mapping)), 1.0)
            self.assertLess(_reward(dst, dict(oracle, episode_id=moved["public"]["manifest"]["episode_id"])), 1.0)

    def test_prefixed_ids_follow_the_table(self):
        dst = self.tmp / "p"
        transform(self.src, dst, 3, id_style="prefixed")
        public = load_episode(dst)["public"]
        self.assertTrue(all(row["row_id"].startswith("row_") for row in public["ledger"]))
        self.assertTrue(all(doc["doc_id"].startswith("doc_") for doc in public["invoices"] + public["payments"]))

    def test_no_shuffle_no_rename_keeps_records(self):
        dst = self.tmp / "same"
        transform(self.src, dst, 3, shuffle=False, rename=False)
        self.assertEqual(load_episode(dst), self.episode)

    def test_keep_episode_id(self):
        dst = self.tmp / "keep"
        document = transform(self.src, dst, 3, keep_episode_id=True)
        self.assertEqual(document["episode_id"]["old"], document["episode_id"]["new"])

    def test_tiny_fixture_and_optional_pre_documents(self):
        src = self.tmp / "tiny"
        shutil.copytree(TINY, src)
        # The generator is adding pre_invoices/pre_payments (CONTRACTS v2): ids inside must be renamed too.
        for table in ("invoices", "payments"):
            shutil.copy(src / "public" / f"{table}.jsonl", src / "private" / f"pre_{table}.jsonl")
        dst = self.tmp / "tiny_out"
        document = transform(src, dst, 1, id_style="hex")
        for table in ("invoices", "payments"):
            def rows(relative):
                lines = (dst / relative).read_text(encoding="utf-8").splitlines()
                return sorted((json.loads(line)["doc_id"] for line in lines if line.strip()))
            pre = rows(f"private/pre_{table}.jsonl")
            self.assertEqual(pre, rows(f"public/{table}.jsonl"))
            self.assertTrue(set(pre) <= set(document["ids"].values()))
        oracle = metamorph.oracle_answer(load_episode(TINY))
        self.assertEqual(_reward(dst, metamorph.map_answer(oracle, load_mapping(dst))), _reward(TINY, oracle))

    def test_refuses_non_empty_destination_and_missing_files(self):
        (self.tmp / "full").mkdir()
        (self.tmp / "full" / "x").write_text("x")
        with self.assertRaises(TransformError):
            transform(self.src, self.tmp / "full", 1)
        with self.assertRaises(TransformError):
            transform(self.tmp / "nowhere", self.tmp / "out", 1)

    def test_transform_does_not_import_the_generator(self):
        tree = ast.parse((ROOT / "tools" / "transform.py").read_text(encoding="utf-8"))
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        modules |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual({m for m in modules if m and m.startswith("auditgym")}, {"auditgym.rng"})


class MetamorphTest(unittest.TestCase):
    def test_pairs_pass(self):
        out = io.StringIO()
        summary = metamorph.run(3, seed_base=777, out=out)
        self.assertEqual(summary["failures"], [], out.getvalue())
        self.assertEqual(summary["results"]["transform"], [3, 3])

    def test_surface_mapping_is_needed(self):
        """Sensitivity: without the id table, the oracle of one surface scores nothing on another."""
        first = generate(5, CONFIG)
        second = generate(5, dict(CONFIG, surface_seed=99))
        evaluator = Evaluator(clock=FakeClock())
        evaluator.reset(5, dict(CONFIG, surface_seed=99))
        unmapped = dict(metamorph.oracle_answer(first), episode_id=second["public"]["manifest"]["episode_id"])
        self.assertLess(evaluator.step({"action": "submit", "findings": unmapped})[1], 1.0)


if __name__ == "__main__":
    unittest.main()
