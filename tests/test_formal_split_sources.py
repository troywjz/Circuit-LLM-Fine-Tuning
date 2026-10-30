import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "formal_split_sources.py"
SPEC = importlib.util.spec_from_file_location("formal_split_sources", SCRIPT)
formal_split_sources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formal_split_sources)


def row(source_id, project_id, group_id, question_id, title, **extra):
    return {
        "source_id": source_id,
        "project_id": project_id,
        "group_id": group_id,
        "question_id": question_id,
        "question_text": title,
        "answer_text": "Answer for " + source_id,
        "isolation_status": "isolated",
        "training_approved": False,
        "raw_record": {"question": {"title": title}, "duplicate_question_ids": []},
        **extra,
    }


class BuildPlanTests(unittest.TestCase):
    def plan(self, rows, reserved=None, seed=7):
        return formal_split_sources.build_plan(rows, reserved or {
            "status": "proposed_reserved_sources_not_final_cases", "sources": []
        }, seed)

    def test_shared_project_or_group_never_crosses_splits(self):
        rows = [row("a", "p", "g1", "q1", "Alpha topic"),
                row("b", "p", "g2", "q2", "Beta topic"),
                row("c", "other", "g3", "q3", "Gamma topic"),
                row("d", "different", "g3", "q4", "Delta topic")]
        planned, _ = self.plan(rows)
        by_id = {r["source_id"]: r for r in planned}
        self.assertEqual(by_id["a"]["split_candidate"], by_id["b"]["split_candidate"])
        self.assertEqual(by_id["c"]["split_candidate"], by_id["d"]["split_candidate"])

    def test_duplicate_and_similar_titles_are_family_grouped(self):
        rows = [row("a", "p1", "g1", "q1", "How to size a buck converter?"),
                row("b", "p2", "g2", "q2", "How do I size a buck converter?"),
                row("c", "p3", "g3", "q3", "Unrelated topic", raw_record={
                    "question": {"title": "Unrelated topic"},
                    "duplicate_question_ids": ["q1"]})]
        planned, _ = self.plan(rows)
        fam = {r["source_id"]: r["family_id"] for r in planned}
        self.assertEqual(fam["a"], fam["b"])
        self.assertEqual(fam["a"], fam["c"])

    def test_reserved_match_excludes_entire_family(self):
        rows = [row("a", "p", "g", "q1", "Exact reserved title"),
                row("b", "p", "g", "q2", "Sibling title")]
        reserved = {"status": "proposed_reserved_sources_not_final_cases", "sources": [
            {"source_id": "old", "group_id": "x", "question_id": "q1",
             "title_original": "Exact reserved title"}]}
        planned, _ = self.plan(rows, reserved)
        self.assertTrue(all(r["split_candidate"] == "excluded_pending_review" for r in planned))

    def test_reserved_title_match_with_distinct_ids_is_isolated(self):
        # 旧来源只有标题相似时也要隔离，不能把保留标题当作并查集下标。
        rows = [row("new", "project-new", "group-new", "q-new", "How to size a buck converter?")]
        reserved = {"status": "proposed_reserved_sources_not_final_cases", "sources": [
            {"source_id": "old", "group_id": "group-old", "question_id": "q-old",
             "title_original": "How to size a buck converter?"}]}
        planned, _ = self.plan(rows, reserved)
        self.assertEqual(planned[0]["split_candidate"], "excluded_pending_review")

    def test_duplicate_id_to_old_question_is_isolated(self):
        rows = [row("new", "project-new", "group-new", "q-new", "Another topic",
                    raw_record={"question": {"title": "Another topic"},
                                "duplicate_question_ids": ["old-question"]})]
        reserved = {"status": "proposed_reserved_sources_not_final_cases",
                    "excluded_question_ids": ["old-question"], "sources": []}
        planned, _ = self.plan(rows, reserved)
        self.assertEqual(planned[0]["split_candidate"], "excluded_pending_review")

    def test_media_review_family_is_pending(self):
        rows = [row("a", "p", "g", "q1", "Alpha", source_media_review={"status": "pending"}),
                row("b", "p", "g", "q2", "Beta")]
        planned, _ = self.plan(rows)
        self.assertTrue(all(r["split_candidate"] == "excluded_pending_review" for r in planned))

    def test_seed_repeats_deterministically_and_summary_counts(self):
        rows = [row(str(i), "p" + str(i), "g" + str(i), "q" + str(i), "Topic " + str(i)) for i in range(30)]
        first = self.plan(rows, seed=42)
        second = self.plan(rows, seed=42)
        self.assertEqual(first, second)
        planned, summary = first
        self.assertTrue(all(k in r for r in planned for k in ("family_id", "split_candidate", "split_status")))
        self.assertTrue(any(isinstance(v, int) for v in summary.values()))


class OutputPathTests(unittest.TestCase):
    def test_private_new_target_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private"
            root.mkdir()
            self.assertTrue(formal_split_sources.validate_output_path(root / "result.json", root))

    def test_outside_or_existing_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private"
            root.mkdir()
            outside = Path(tmp) / "outside.json"
            with self.assertRaises((ValueError, OSError)):
                formal_split_sources.validate_output_path(outside, root)
            existing = root / "exists.json"
            existing.touch()
            with self.assertRaises((ValueError, FileExistsError, OSError)):
                formal_split_sources.validate_output_path(existing, root)


if __name__ == "__main__":
    unittest.main()
