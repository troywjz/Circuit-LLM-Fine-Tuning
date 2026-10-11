"""Demo 来源筛选与元数据隔离的轻量回归检查。"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_demo_data.py"
spec = importlib.util.spec_from_file_location("build_demo_data", SCRIPT)
demo = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(demo)


def source(sid="se-question-1", gid="se-group-1", qid=1, title="I2C pull-up resistor", body="How should I select a pull-up resistor for an I2C bus?", answer="Use rise time and sink current limits."):
    return {"source_id":sid,"group_id":gid,"project_id":"se-1","topic":"i2c","risk_review_tags":[],"needs_media":False,"training_eligible":False,"duplicate_question_ids":[],"question":{"question_id":qid,"title":title,"body_text":body,"post_url":"https://example.invalid/q","author":"author","content_license":"CC BY-SA 4.0"},"answer":{"body_text":answer,"post_url":"https://example.invalid/a","author":"answerer","content_license":"CC BY-SA 4.0"},"sources":[{"source_version":{"snapshot_sha256":"abc"}}]}


class DemoSourceTests(unittest.TestCase):
    def test_reserved_source_or_group_is_excluded(self):
        blocked=(set(),{"se-group-1"},set(),[])
        self.assertEqual(demo.is_candidate(source(),blocked),(False,"reserved-lineage"))

    def test_safety_and_missing_context_rules(self):
        blocked=(set(),set(),set(),[])
        risky=source(body="Can I use a 230 V mains supply here?")
        self.assertEqual(demo.is_candidate(risky,blocked),(False,"safety-topic"))
        incomplete=source(body=" ")
        self.assertEqual(demo.is_candidate(incomplete,blocked),(False,"missing-content"))

    def test_prepared_row_keeps_origin_metadata_outside_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"rows.jsonl"
            demo._write_sourcepack(path,[source()],"train")
            row=json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["review"]["approved_for_demo"],False)
        self.assertEqual(row["sources"][0]["question_license"],"CC BY-SA 4.0")
        self.assertNotIn("messages",row)
        self.assertEqual(row["group_id"],"se-group-1")

    def test_general_scenarios_do_not_claim_teacher_output_license(self):
        item=demo._general_case(1)
        self.assertEqual(item["sources"][0]["license"],"MIT")
        self.assertIn("不自动主张MIT",item["sources"][0]["rights_note"])
        self.assertFalse(item["review"]["approved_for_demo"])
        scenarios=[demo._general_case(i)["scenario"] for i in range(1,21)]
        self.assertEqual(len(set(scenarios)),20)
        self.assertGreaterEqual(len({s.split("：")[0].split("，")[0] for s in scenarios}),4)

    def test_duplicate_question_lineage_is_grouped_before_split(self):
        a=source("se-question-a","se-group-a",11)
        b=source("se-question-b","se-group-b",12)
        a["duplicate_question_ids"]=[12]
        rows=[a,b]
        demo._assign_duplicate_families(rows)
        self.assertEqual(rows[0]["_split_family"],rows[1]["_split_family"])

    def test_failure_log_keeps_only_safe_diagnostic_fields(self):
        class FakeTeacherError(Exception):
            http_status=429
            category="rate_limit"
            retry_after_seconds=12.5
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"调用失败.json"
            demo._write_call_failure(path,source_id="se-question-1",row_sha256="a"*64,
                prompt_sha256="b"*64,requested_model="claude-opus-5-5",max_output_tokens=8192,
                error=FakeTeacherError("secret-key https://private.example/private prompt text"),
                completed_count=0,wallet_usd=9.9)
            text=path.read_text(encoding="utf-8")
            row=json.loads(text)
        self.assertEqual(row["http_status"],429)
        self.assertEqual(row["category"],"rate_limit")
        self.assertEqual(row["retry_after_seconds"],12.5)
        self.assertTrue(row["utc"].endswith("Z"))
        self.assertEqual(row["wallet_usd"],9.9)
        for private in ("secret-key","https://private.example","prompt text"):
            self.assertNotIn(private,text)


if __name__ == "__main__":
    unittest.main()
