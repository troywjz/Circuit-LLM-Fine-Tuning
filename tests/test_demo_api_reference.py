"""Claude 强参考 runner 与多轮教师 API 的离线协议测试。"""
from __future__ import annotations

import json
import io
import builtins
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import demo_api_reference as reference
from teacher_api import TeacherConfig, TeacherError, call_teacher_messages


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class FakeOpener:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        return FakeResponse(self.payload)


def config(provider="openai", api="responses"):
    return TeacherConfig(provider, "http://127.0.0.1:8123/gateway/v1", "unit-test-only-key",
                         "claude-opus-5-5", api, 8192, 1, 0)


class TeacherMessagesTests(unittest.TestCase):
    def test_protocol_preserves_role_messages_and_system(self):
        messages = [{"role": "system", "content": "neutral system"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "own answer"},
                    {"role": "user", "content": "correction"}]
        for cfg in (config(), config(api="chat_completions"), config("anthropic")):
            payload = ({"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]}
                       if cfg.provider == "openai" and cfg.openai_api == "responses" else
                       {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
                       if cfg.provider == "openai" else
                       {"content": [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "ok"}], "stop_reason": "end_turn"})
            payload["model"] = "actual-model-r1"
            opener = FakeOpener(payload)
            with self.subTest(provider=cfg.provider, api=cfg.openai_api), patch("teacher_api.urllib.request.build_opener", return_value=opener):
                result = call_teacher_messages(cfg, messages)
                self.assertEqual(result["text"], "ok")
                self.assertEqual(result["reported_model"], "actual-model-r1")
                self.assertEqual(result["model"], "actual-model-r1")
            body = json.loads(opener.requests[0].data)
            if cfg.provider == "openai" and cfg.openai_api == "responses":
                self.assertEqual(body["instructions"], "neutral system")
                self.assertEqual(body["input"], messages[1:])
                self.assertEqual(body["max_output_tokens"], 8192)
                self.assertFalse(body["store"])
                self.assertFalse(body["stream"])
            elif cfg.provider == "openai":
                self.assertEqual(body["messages"], messages)
            else:
                self.assertEqual(body["system"], "neutral system")
                self.assertEqual(body["messages"], messages[1:])

    def test_rejects_invalid_role_history(self):
        with self.assertRaises(TeacherError):
            call_teacher_messages(config(), [{"role": "user", "content": "q"},
                                             {"role": "user", "content": "another"}])

    def test_http_error_categories_are_sanitized_and_retries_are_bounded(self):
        cases = [
            (429, {"code": "rate_limit_exceeded", "message": "private message https://private.invalid/key"},
             "rate_limit", "教师接口限流", 3, 300.0),
            (429, {"type": "insufficient_quota", "message": "billing account 12345"},
             "quota_billing", "教师接口额度或账单受限", 1, 300.0),
            (401, {"type": "authentication_error", "message": "sk-secret-private"},
             "auth", "教师接口认证或权限失败", 1, None),
            (429, {"message": "opaque private diagnostic"}, "unknown", "教师接口请求失败", 3, 300.0),
        ]
        for status, detail, category, label, expected_calls, retry_seconds in cases:
            body = json.dumps({"error": detail}).encode()
            class ErrorOpener:
                def __init__(self): self.requests = []
                def open(self, request, timeout):
                    self.requests.append(request)
                    # Each bounded retry gets a fresh error body.
                    raise urllib.error.HTTPError("https://private.invalid/path", status, "secret-reason",
                        {"Retry-After": "999"}, io.BytesIO(body))
            opener = ErrorOpener()
            cfg = config()
            cfg.max_retries = 2
            with self.subTest(category=category), patch("teacher_api.urllib.request.build_opener", return_value=opener), patch("teacher_api.time.sleep"):
                with self.assertRaises(TeacherError) as caught:
                    call_teacher_messages(cfg, [{"role": "user", "content": "q"}])
            self.assertEqual(len(opener.requests), expected_calls)
            self.assertEqual(caught.exception.http_status, status)
            self.assertEqual(caught.exception.category, category)
            self.assertEqual(caught.exception.retry_after_seconds, retry_seconds)
            self.assertIn(label, str(caught.exception))
            for secret in ("private.invalid", "private message", "private diagnostic", "sk-secret-private", "12345"):
                self.assertNotIn(secret, str(caught.exception))


class ReferenceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cases = self.root / "cases.jsonl"
        self.output = self.root / "out.jsonl"
        self.budget = self.root / "budget.json"
        self.cases.write_text(json.dumps({"case_id": "c1", "kind": "general", "split": "test",
            "input_messages": [{"role": "user", "content": "Q"}], "user_turns": ["revision"],
            "gold": "SECRET GOLD", "rubric": "PRIVATE RUBRIC", "source_url": "PRIVATE SOURCE"}) + "\n", encoding="utf-8")
        self.cfg = config()

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, *extra):
        return ["--cases", str(self.cases), "--output", str(self.output),
                "--env-file", str(self.root / "missing.env"), "--budget-state", str(self.budget), *extra]

    def test_dry_run_needs_no_credentials_or_network(self):
        with patch("teacher_api.load_config", side_effect=AssertionError("credentials loaded")), patch.object(reference, "_wallet", side_effect=AssertionError("network")):
            self.assertEqual(reference.main(self._args("--dry-run")), 0)

    def test_partial_resume_reuses_turn_and_drops_gold_metadata(self):
        calls = []
        def fake_call(cfg, messages):
            calls.append(messages)
            if len(calls) == 2:
                raise TeacherError("教师接口请求失败（HTTP 401）。")
            return {"text": "answer", "usage": {"input_tokens": 2, "output_tokens": 3},
                    "finish_reason": "end_turn", "reported_model": "service-model-v2"}
        real_import = builtins.__import__
        def reject_local_model_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"qwen35_common", "unsloth", "torch"}:
                raise AssertionError("API reference runner attempted to import local inference modules")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=reject_local_model_import), \
             patch("teacher_api.load_config", return_value=self.cfg), \
             patch.object(reference, "_wallet", return_value=8.0), \
             patch("teacher_api.call_teacher_messages", side_effect=fake_call):
            self.assertEqual(reference.main(self._args()), 2)
            self.assertEqual(reference.main(self._args("--resume")), 0)
        self.assertEqual(len(calls), 3)  # 第一轮不重复收费，只补齐两轮中的第二轮。
        self.assertNotIn("SECRET GOLD", json.dumps(calls, ensure_ascii=False))
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["answers_by_turn"], ["answer", "answer"])
        self.assertEqual(result["requested_model"], "claude-opus-5-5")
        self.assertEqual(result["reported_model"], "service-model-v2")
        self.assertEqual(result["reported_models"], ["service-model-v2", "service-model-v2"])
        self.assertEqual([t["reported_model"] for t in result["turns"]], ["service-model-v2", "service-model-v2"])
        self.assertTrue(all(t["service_default_decoding"] for t in result["turns"]))
        self.assertEqual(result["full_dialogue"][-2:], [{"role": "user", "content": "revision"}, {"role": "assistant", "content": "answer"}])

    def test_resume_rejects_changed_case_fingerprint(self):
        with patch("teacher_api.load_config", return_value=self.cfg), patch.object(reference, "_wallet", return_value=8.0), patch("teacher_api.call_teacher_messages", return_value={"text": "answer", "usage": {}, "finish_reason": "end_turn"}):
            self.assertEqual(reference.main(self._args()), 0)
            self.cases.write_text(self.cases.read_text(encoding="utf-8").replace('"Q"', '"Q2"'), encoding="utf-8")
            self.assertEqual(reference.main(self._args("--resume")), 2)

    def test_low_balance_stops_before_reference_request(self):
        with patch("teacher_api.load_config", return_value=self.cfg), patch.object(reference, "_wallet", return_value=0.49), patch("teacher_api.call_teacher_messages") as caller:
            self.assertEqual(reference.main(self._args()), 2)
        caller.assert_not_called()

    def test_tool_extra_body_is_rejected_before_wallet_or_model_request(self):
        for field in ("tools", "functions", "function_call", "parallel_tool_calls"):
            self.cfg.extra_body = {field: [{"type": "function"}]}
            with self.subTest(field=field), \
                 patch("teacher_api.load_config", return_value=self.cfg), \
                 patch.object(reference, "_wallet") as wallet, \
                 patch("teacher_api.call_teacher_messages") as caller:
                self.assertEqual(reference.main(self._args()), 2)
            wallet.assert_not_called()
            caller.assert_not_called()

    def test_fingerprint_includes_extra_body_and_timeout(self):
        args = SimpleNamespace(run_label="claude-opus-5-5", step0=0, max_output_tokens=8192)
        cfg = config()
        original = reference._fingerprint(cfg, args, "cases-hash")
        cfg.extra_body = {"temperature": 0.1}
        self.assertNotEqual(reference._fingerprint(cfg, args, "cases-hash"), original)
        cfg.extra_body = {}
        cfg.timeout_seconds += 1
        self.assertNotEqual(reference._fingerprint(cfg, args, "cases-hash"), original)


if __name__ == "__main__":
    unittest.main()
