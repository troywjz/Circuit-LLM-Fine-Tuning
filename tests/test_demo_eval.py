import json
import tempfile
import unittest
from pathlib import Path

from scripts.demo_eval_tools import aggregate, plot, prepare_blind, read_jsonl


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def cases():
    return [
        {"case_id": "D01", "kind": "domain", "split": "dev", "scoring": {"criteria": [
            {"dimension": "任务与约束", "applicable": True}, {"dimension": "事实正确性", "applicable": True}], "hard_errors": []}},
        {"case_id": "D02", "kind": "domain", "split": "dev", "scoring": {"criteria": [
            {"dimension": "任务与约束", "applicable": True}, {"dimension": "事实正确性", "applicable": True}], "hard_errors": []}},
    ]


def score_rows(blind_rows):
    return [{"run_id": row["run_id"], "case_id": row["case_id"],
             "scores": {"任务与约束": 2, "事实正确性": 2}, "hard_errors": []} for row in blind_rows]


class DemoEvalTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.tmp_path = Path(self.temp.name)

  def test_blind_allows_same_case_for_separate_checkpoints(self):
    tmp_path = self.tmp_path
    raw = tmp_path / "answers.jsonl"
    blind = tmp_path / "blind.jsonl"
    mapping = tmp_path / "mapping.jsonl"
    write_jsonl(raw, [
        {"case_id": "D01", "model": "ft", "step": 15, "answer": "first checkpoint"},
        {"case_id": "D01", "model": "ft", "step": 60, "answer": "later checkpoint"},
    ])
    prepare_blind(raw, blind, mapping, seed="repeatable-order")
    self.assertEqual(len(read_jsonl(blind)), 2)
    mapped = read_jsonl(mapping)
    self.assertEqual({(row["model"], row["step"]) for row in mapped}, {("ft", 15), ("ft", 60)})
    self.assertEqual(len({row["run_id"] for row in mapped}), 2)


  def test_aggregate_scores_only_complete_equal_case_sets(self):
    tmp_path = self.tmp_path
    casefile, raw, blind, mapping, ratings, output = [tmp_path / name for name in
        ("cases.jsonl", "raw.jsonl", "blind.jsonl", "mapping.jsonl", "ratings.jsonl", "aggregate.jsonl")]
    write_jsonl(casefile, cases())
    rows = []
    for model, step in (("base", 0), ("ft", 15), ("ft", 30), ("ft", 60), ("claude-opus-5-5", 0)):
        rows.extend({"case_id": case_id, "model": model, "step": step, "answer": f"{model}-{step}-{case_id}"}
                    for case_id in ("D01", "D02"))
    write_jsonl(raw, rows)
    prepare_blind(raw, blind, mapping, seed="fixed")
    write_jsonl(ratings, score_rows(read_jsonl(blind)))
    aggregate(casefile, blind, ratings, mapping, output)
    scored = read_jsonl(output)
    self.assertEqual(len(scored), 10)
    self.assertTrue(all(row["score"] == 100 and row["pass"] for row in scored))
    self.assertEqual({row["model"] for row in scored}, {"base", "ft", "claude-opus-5-5"})


  def test_aggregate_rejects_incomplete_model_case_set(self):
    tmp_path = self.tmp_path
    casefile, raw, blind, mapping, ratings = [tmp_path / name for name in
        ("cases.jsonl", "raw.jsonl", "blind.jsonl", "mapping.jsonl", "ratings.jsonl")]
    write_jsonl(casefile, cases())
    write_jsonl(raw, [
        {"case_id": "D01", "model": "base", "step": 0, "answer": "one"},
        {"case_id": "D01", "model": "ft", "step": 15, "answer": "one"},
        {"case_id": "D02", "model": "ft", "step": 15, "answer": "two"},
    ])
    prepare_blind(raw, blind, mapping, seed="fixed")
    write_jsonl(ratings, score_rows(read_jsonl(blind)))
    with self.assertRaisesRegex(ValueError, "incomplete case set"):
        aggregate(casefile, blind, ratings, mapping, tmp_path / "out.jsonl")


  def test_aggregate_rejects_unknown_dimensions_and_invalid_hard_errors(self):
    tmp_path = self.tmp_path
    casefile, raw, blind, mapping = [tmp_path / name for name in
        ("cases.jsonl", "raw.jsonl", "blind.jsonl", "mapping.jsonl")]
    write_jsonl(casefile, [{"case_id": "D01", "kind": "domain", "split": "dev", "scoring": {
        "criteria": [{"dimension": "invented", "applicable": True}], "hard_errors": []}}])
    write_jsonl(raw, [{"case_id": "D01", "model": "base", "step": 0, "answer": "text"}])
    prepare_blind(raw, blind, mapping)
    blind_row = read_jsonl(blind)[0]
    write_jsonl(tmp_path / "ratings.jsonl", [{"run_id": blind_row["run_id"], "case_id": "D01", "scores": {"invented": 2}, "hard_errors": []}])
    with self.assertRaisesRegex(ValueError, "unknown or repeated rubric dimension"):
        aggregate(casefile, blind, tmp_path / "ratings.jsonl", mapping, tmp_path / "out.jsonl")
    write_jsonl(casefile, cases()[:1])
    write_jsonl(tmp_path / "ratings.jsonl", [{"run_id": blind_row["run_id"], "case_id": "D01",
        "scores": {"任务与约束": 2, "事实正确性": 2}, "hard_errors": [None]}])
    with self.assertRaisesRegex(ValueError, "hard_errors must be"):
        aggregate(casefile, blind, tmp_path / "ratings.jsonl", mapping, tmp_path / "out.jsonl")


  def test_plot_refuses_empty_scores_instead_of_drawing_fake_zeros(self):
    tmp_path = self.tmp_path
    empty = tmp_path / "aggregate.jsonl"
    empty.write_text("", encoding="utf-8")
    with self.assertRaisesRegex(ValueError, "no domain dev scores"):
        plot(empty, tmp_path / "missing-loss.csv", tmp_path / "curve", "test")
    self.assertFalse((tmp_path / "curve.png").exists())
