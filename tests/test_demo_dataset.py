"""Demo 来源筛选与元数据隔离的轻量回归检查。"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
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


class CandidateParsingTests(unittest.TestCase):
    def setUp(self):
        self.row={"sample_id":"demo-train-se-question-1","source_id":"se-question-1","group_id":"se-group-1","split":"train","task":"给出方案与取舍","sources":[]}
        self.payload={"messages":[{"role":"user","content":"如何选择？"},{"role":"assistant","content":"先确认具体条件，再核对器件资料。"}],"used_source_ids":["se-question-1"]}

    def test_complete_json_fence_is_accepted_and_recorded(self):
        text="```json\n"+json.dumps(self.payload,ensure_ascii=False)+"\n```"
        with patch.object(demo,"_token_count",return_value=32):
            candidate=demo._candidate_from_reply(self.row,text,"teacher",{"output_tokens":12})
        self.assertEqual(candidate["messages"],self.payload["messages"])
        self.assertIn("移除完整外层 JSON 代码围栏",candidate["modifications"])
        self.assertFalse(candidate["review"]["approved_for_demo"])

    def test_json_fence_surrounded_by_prose_is_rejected(self):
        text="结果如下：\n```json\n"+json.dumps(self.payload,ensure_ascii=False)+"\n```"
        with self.assertRaisesRegex(demo.BuildError,"不是 JSON"):
            demo._candidate_from_reply(self.row,text,"teacher",{})

    def test_unknown_dangerous_field_remains_rejected(self):
        payload={**self.payload,"debug":"hidden chain of thought"}
        with self.assertRaisesRegex(demo.BuildError,"字段无效"):
            demo._candidate_from_reply(self.row,json.dumps(payload,ensure_ascii=False),"teacher",{})


class ReviewNoteTests(unittest.TestCase):
    def setUp(self):
        self.row={"sample_id":"demo-train-se-question-1","source_id":"se-question-1","group_id":"se-group-1","split":"train","topic":"i2c","task":"给出方案与取舍","title_original":"I2C pull-up","question_original":"How to select?","answer_reference_original":"Use constraints.","sources":[]}
        self.notes={"topic_notes":{"i2c":"I2C_TOPIC_RULE","adc":"OTHER_TOPIC_RULE"},"common_note":"COMMON_RULE","sources":[{"title":"review metadata","url":"https://private.example/source"}]}

    def test_prompt_only_includes_current_topic_and_common_note(self):
        prompt=demo._teacher_prompt(self.row,self.notes)
        self.assertIn("COMMON_RULE",prompt)
        self.assertIn("I2C_TOPIC_RULE",prompt)
        self.assertNotIn("OTHER_TOPIC_RULE",prompt)
        self.assertNotIn("https://private.example/source",prompt)

    def test_note_hash_and_fingerprint_change_without_hashing_api_key(self):
        config=SimpleNamespace(provider="test",base_url="https://api.example/v1",model_id="model",openai_api="responses",api_key="secret-one")
        first_hash=demo._review_note_hash(self.row,self.notes)
        changed={**self.notes,"topic_notes":{"i2c":"CHANGED_RULE","adc":"OTHER_TOPIC_RULE"}}
        second_hash=demo._review_note_hash(self.row,changed)
        self.assertNotEqual(first_hash,second_hash)
        self.assertNotEqual(demo._config_fingerprint(config,first_hash),demo._config_fingerprint(config,second_hash))
        same_hash_config=SimpleNamespace(**{**vars(config),"api_key":"secret-two"})
        self.assertEqual(demo._config_fingerprint(config,first_hash),demo._config_fingerprint(same_hash_config,first_hash))

    def test_resume_reuses_legacy_raw_without_calling_teacher(self):
        class FakeTeacherError(Exception):
            pass
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            input_path=root/"input.jsonl"
            approval_path=root/"approval.json"
            outdir=root/"out"
            outdir.mkdir()
            input_path.write_text(json.dumps(self.row,ensure_ascii=False)+"\n",encoding="utf-8")
            approval_path.write_text(json.dumps({"reviewer":"main","status":"approved_for_teacher_generation","source_ids":[self.row["source_id"]]}),encoding="utf-8")
            balance=10.0
            (outdir/".teacher_budget.json").write_text(json.dumps({"starting_balance":balance,"cap_usd":10.0,"reserve_usd":2.0}),encoding="utf-8")
            raw_text=json.dumps({"messages":[{"role":"user","content":"How?"},{"role":"assistant","content":"Check the conditions first."}],"used_source_ids":[self.row["source_id"]]},ensure_ascii=False)
            config=SimpleNamespace(provider="test",base_url="https://api.example/v1",model_id="model",openai_api="responses",api_key="unused",max_retries=3,max_output_tokens=1024)
            prompt_hash=demo._sha(demo.SYSTEM_PROMPT+"\0"+demo._teacher_prompt(self.row))
            raw={"status":"success","source_id":self.row["source_id"],"row_sha256":demo._sha(demo._canonical(self.row)),"prompt_sha256":prompt_hash,"config_fingerprint":demo._config_fingerprint(config),"raw_text":raw_text,"model":"test","usage":{"output_tokens":8}}
            raw_path=outdir/"teacher_raw"/(self.row["source_id"]+".json")
            raw_path.parent.mkdir()
            raw_path.write_text(json.dumps(raw,ensure_ascii=False),encoding="utf-8")
            before=raw_path.read_bytes()
            api=ModuleType("teacher_api")
            api.TeacherError=FakeTeacherError
            api.call_teacher=lambda *args: self.fail("resume must not call Teacher")
            api.load_config=lambda path: config
            args=SimpleNamespace(input=input_path,approval_file=approval_path,env_file=root/".env",outdir=outdir,limit=1,dry_run=False,resume=True)
            with patch.dict(sys.modules,{"teacher_api":api}),patch.object(demo,"_wallet",return_value=balance),patch.object(demo,"_ensure_tokenizer"),patch.object(demo,"_token_count",return_value=20):
                self.assertEqual(demo.generate(args),0)
            self.assertEqual(raw_path.read_bytes(),before)
            candidate=json.loads((outdir/"train_candidates.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(candidate["review_note_hash"],None)
            self.assertEqual(candidate["row_hash"],raw["row_sha256"])
            self.assertEqual(candidate["prompt_hash"],prompt_hash)
            self.assertEqual(candidate["config_fingerprint"],raw["config_fingerprint"])


if __name__ == "__main__":
    unittest.main()
