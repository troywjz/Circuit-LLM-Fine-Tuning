"""Teacher 接口与数据整理流程的离线契约测试，不访问真实服务。"""

from __future__ import annotations

import json
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from teacher_api import TeacherConfig, TeacherError, call_teacher, endpoint_url, load_config
import prepare_dataset as prep


FAKE_KEY = "unit-test-only-key"


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200, headers: dict | None = None):
        self.payload = payload
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class FakeOpener:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error:
            raise self.error
        return self.response


def config(provider="openai", api="responses", base="http://127.0.0.1:8123/gateway"):
    return TeacherConfig(provider, base, FAKE_KEY, "demo-model", api, 128, 1, 0)


class TeacherApiTests(unittest.TestCase):
    def test_endpoint_preserves_prefix_and_appends_once(self):
        cfg = config()
        self.assertEqual(endpoint_url(cfg), "http://127.0.0.1:8123/gateway/responses")
        cfg.base_url += "/v1"
        self.assertEqual(endpoint_url(cfg), "http://127.0.0.1:8123/gateway/v1/responses")
        self.assertEqual(endpoint_url(config("openai", "chat_completions")),
                         "http://127.0.0.1:8123/gateway/chat/completions")
        self.assertEqual(endpoint_url(config(base="http://127.0.0.1:8123")),
                         "http://127.0.0.1:8123/v1/responses")
        self.assertEqual(endpoint_url(config("anthropic")),
                         "http://127.0.0.1:8123/gateway/messages")

    def test_protocol_request_shapes_and_text_parsing(self):
        cases = [
            (config(), {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]}, "authorization", "ok"),
            (config(api="chat_completions"), {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}, "Bearer", "ok"),
            (config("anthropic"), {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}, "x-api-key", "ok"),
        ]
        for cfg, payload, expected_header, expected_text in cases:
            with self.subTest(provider=cfg.provider, api=cfg.openai_api):
                opener = FakeOpener(FakeResponse(payload))
                with patch("teacher_api.urllib.request.build_opener", return_value=opener):
                    result = call_teacher(cfg, "system", "question")
                self.assertEqual(result["text"], expected_text)
                req = opener.requests[0][0]
                body = json.loads(req.data)
                self.assertIn(expected_header.casefold(), repr(req.header_items()).casefold())
                if cfg.openai_api == "responses" and cfg.provider == "openai":
                    self.assertEqual(body["instructions"], "system")
                    self.assertEqual(body["input"], "question")
                    self.assertFalse(body["store"])
                elif cfg.provider == "openai":
                    self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
                else:
                    self.assertEqual(body["system"], "system")

    def test_truncation_refusal_and_tool_only_results_are_rejected(self):
        payloads = [
            {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
            {"output": [{"type": "refusal", "content": []}]},
            {"choices": [{"message": {"tool_calls": [{"id": "x"}]}, "finish_reason": "tool_calls"}]},
        ]
        cfgs = [config(api="chat_completions"), config(), config(api="chat_completions")]
        for cfg, payload in zip(cfgs, payloads):
            with self.subTest(payload=payload):
                with patch("teacher_api.urllib.request.build_opener", return_value=FakeOpener(FakeResponse(payload))):
                    with self.assertRaises(TeacherError):
                        call_teacher(cfg, "s", "u")

    def test_anthropic_thinking_is_ignored_and_only_normal_text_end_is_accepted(self):
        cfg = config("anthropic")
        payload = {"content": [{"type": "thinking", "thinking": "private reasoning"},
                               {"type": "text", "text": "final answer"}],
                   "stop_reason": "end_turn"}
        with patch("teacher_api.urllib.request.build_opener", return_value=FakeOpener(FakeResponse(payload))):
            result = call_teacher(cfg, "s", "u")
        self.assertEqual(result["text"], "final answer")
        self.assertNotIn("private reasoning", json.dumps(result))
        for stop in ("tool_use", "unknown", None):
            bad = {"content": [{"type": "text", "text": "answer"}], "stop_reason": stop}
            with self.subTest(stop_reason=stop), patch("teacher_api.urllib.request.build_opener", return_value=FakeOpener(FakeResponse(bad))):
                with self.assertRaises(TeacherError):
                    call_teacher(cfg, "s", "u")

    def test_credentials_hidden_and_auth_errors_do_not_leak_or_retry(self):
        self.assertNotIn(FAKE_KEY, repr(config()))
        import urllib.error
        opener = FakeOpener(error=urllib.error.HTTPError("http://localhost", 401, "denied", {}, None))
        cfg = config()
        cfg.max_retries = 2
        with patch("teacher_api.urllib.request.build_opener", return_value=opener):
            with self.assertRaises(TeacherError) as caught:
                call_teacher(cfg, "s", "u")
        self.assertNotIn(FAKE_KEY, str(caught.exception))
        self.assertEqual(len(opener.requests), 1)

    def test_config_validation_and_template_placeholders(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.env"
            path.write_text("TEACHER_PROVIDER=openai\nTEACHER_API_KEY=\nTEACHER_BASE_URL=\nTEACHER_MODEL_ID=\n", encoding="utf-8")
            cfg = load_config(path, require_credentials=False)
            self.assertEqual(cfg.provider, "openai")
            path.write_text("TEACHER_PROVIDER=bogus\n", encoding="utf-8")
            with self.assertRaises(TeacherError):
                load_config(path, require_credentials=False)
            path.write_text("TEACHER_BASE_URL=https://user:secret@example.com\n", encoding="utf-8")
            with self.assertRaises(TeacherError) as caught:
                load_config(path, require_credentials=False)
            self.assertNotIn("secret", str(caught.exception))


def sample_row(split="train", group="group-a"):
    return {"source_id": "row-a", "project_id": "project-a", "group_id": group,
            "split": split, "task": "已有设计解释", "student_context": "解释示例电路。",
            "sources": [{"source_id": "doc-a", "content": "自编示例资料。",
                         "url": "https://example.invalid/a", "license": "MIT (自编示例)"}]}


class PreparationValidationTests(unittest.TestCase):
    def test_token_count_uses_single_sample_token_length_for_supported_encodings(self):
        class Encoded(dict):
            @property
            def input_ids(self):
                return self["input_ids"]

        class Tokenizer:
            def __init__(self, result):
                self.result = result

            def apply_chat_template(self, *_args, **_kwargs):
                return self.result

        messages = [{"role": "user", "content": "x"}]
        for result in ([11, 12, 13], {"input_ids": [11, 12, 13]}, Encoded(input_ids=[[11, 12, 13]]), {"input_ids": [[11, 12, 13]]}):
            with self.subTest(result_type=type(result).__name__):
                self.assertEqual(prep.token_count(Tokenizer(result), messages), 3)

    def test_json_and_fenced_json_parse_and_bad_source_reference_rejected(self):
        valid = {"task": "已有设计解释", "messages": [
            {"role": "user", "content": "根据示例资料解释。"},
            {"role": "assistant", "content": "示例回答。"}],
            "evidence": [{"claim": "资料如此说明。", "type": "SOURCE_FACT", "source_ids": ["doc-a"]}],
            "expected_points": ["说明依据"], "hard_errors": []}
        self.assertEqual(prep.parse_teacher_text(json.dumps(valid, ensure_ascii=False)), valid)
        fenced = "```json\n" + json.dumps(valid, ensure_ascii=False) + "\n```"
        self.assertEqual(prep.parse_teacher_text(fenced), valid)
        with self.assertRaises(prep.InputError):
            prep.parse_teacher_text("前缀 " + json.dumps(valid) + " ```")
        valid["evidence"][0]["source_ids"] = ["unknown"]
        with self.assertRaises(prep.InputError):
            prep.validate_generated(valid, sample_row())

    def test_role_sequence_and_split_leakage_are_rejected(self):
        row = sample_row()
        generated = {"task": row["task"], "messages": [
            {"role": "user", "content": "问题"}, {"role": "user", "content": "错误角色"}],
            "evidence": [], "expected_points": [], "hard_errors": []}
        with self.assertRaises(prep.InputError):
            prep.validate_generated(generated, row)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.jsonl"
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
                sample_row("train"), {**sample_row("test", "group-a"), "source_id": "row-b"}]), encoding="utf-8")
            with self.assertRaises(prep.InputError):
                prep.read_input(path, 60000)
            test_leak = sample_row("train", "group-c")
            test_leak["original_split"] = "test"
            path.write_text(json.dumps(test_leak, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(prep.InputError):
                prep.read_input(path, 60000)

    def test_test_gold_is_separate_and_not_eligible_for_training(self):
        row = sample_row("test")
        generated = {"task": row["task"], "messages": [
            {"role": "user", "content": "测试输入"}, {"role": "assistant", "content": "参考答案"}],
            "evidence": [], "expected_points": ["要点"], "hard_errors": []}
        item = prep.normalize(row, generated, "job", "snapshot", {"prompt_hash": "hash", "teacher": {}}, None)
        self.assertEqual(item["input_messages"], generated["messages"][:-1])
        self.assertNotIn("reference_answer", json.dumps(item["input_messages"], ensure_ascii=False))
        self.assertEqual(item["reference_answer"], "参考答案")
        self.assertEqual(item["review"]["status"], "待审")
        self.assertFalse(item["training_eligible"])

    def test_cli_dry_run_needs_no_key_and_does_not_fabricate_candidates(self):
        root = Path(__file__).resolve().parents[1]
        private = root / "本地资料" / "数据准备" / "unittest"
        private.mkdir(parents=True, exist_ok=True)
        source = private / "input.jsonl"
        output = private / "dry-run-output"
        envfile = private / "empty.env"
        source.write_text(json.dumps(sample_row(), ensure_ascii=False), encoding="utf-8")
        envfile.write_text("", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("TEACHER_")}
        env["PYTHONUTF8"] = "1"
        try:
            import subprocess
            result = subprocess.run([sys.executable, str(root / "scripts" / "prepare_dataset.py"),
                                     "--input", str(source), "--env-file", str(envfile),
                                     "--output-dir", str(output), "--dry-run"],
                                    capture_output=True, text=True, encoding="utf-8", env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["dry_run"])
            self.assertFalse((output / "训练候选.jsonl").exists())
        finally:
            import shutil
            shutil.rmtree(private, ignore_errors=True)

    def test_check_config_rejects_fake_placeholders_without_showing_values(self):
        root = Path(__file__).resolve().parents[1]
        private = root / "本地资料" / "数据准备" / "unittest-placeholder"
        private.mkdir(parents=True, exist_ok=True)
        envfile, output = private / "settings.env", private / "out"
        envfile.write_text("TEACHER_PROVIDER=openai\nTEACHER_BASE_URL=https://example.com\n"
                           "TEACHER_API_KEY=your_api_key\nTEACHER_MODEL_ID=replace_me\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("TEACHER_")}
        env["PYTHONUTF8"] = "1"
        try:
            import subprocess
            result = subprocess.run([sys.executable, str(root / "scripts" / "prepare_dataset.py"),
                                     "--env-file", str(envfile), "--output-dir", str(output), "--check-config"],
                                    capture_output=True, text=True, encoding="utf-8", env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("your_api_key", result.stdout + result.stderr)
            self.assertNotIn("replace_me", result.stdout + result.stderr)
            self.assertNotIn("example.com", result.stdout + result.stderr)
        finally:
            import shutil
            shutil.rmtree(private, ignore_errors=True)

    def test_auth_failure_stops_three_job_batch_after_one_call(self):
        root = Path(__file__).resolve().parents[1]
        private = root / "本地资料" / "数据准备" / "unittest-auth"
        private.mkdir(parents=True, exist_ok=True)
        source, envfile, output = private / "input.jsonl", private / "fake.env", private / "out"
        rows = [sample_row()]
        for i in (2, 3):
            row = sample_row(group=f"group-{i}")
            row["source_id"] = f"row-{i}"
            row["sources"][0]["source_id"] = f"doc-{i}"
            rows.append(row)
        source.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        envfile.write_text("TEACHER_PROVIDER=openai\nTEACHER_BASE_URL=http://127.0.0.1:9\n"
                           f"TEACHER_API_KEY={FAKE_KEY}\nTEACHER_MODEL_ID=demo-model\n", encoding="utf-8")
        import urllib.error
        error = urllib.error.HTTPError("http://127.0.0.1:9/v1/responses", 401, "denied", {}, None)
        calls = []

        def fake_call(*_args, **_kwargs):
            calls.append(1)
            raise TeacherError("教师接口请求失败（HTTP 401）。") from error

        argv = ["--input", str(source), "--env-file", str(envfile), "--output-dir", str(output), "--limit", "3"]
        try:
            with patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("TEACHER_")}, clear=True):
                with patch("teacher_api.call_teacher", side_effect=fake_call):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        code = prep.main(argv)
            self.assertEqual(code, 1)
            self.assertEqual(len(calls), 1)
            records = [json.loads(line) for line in (output / "失败记录.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 3, records)
            self.assertEqual(records[1]["status"], "未请求")
            self.assertEqual(records[2]["status"], "未请求")
            self.assertNotIn(FAKE_KEY, (output / "失败记录.jsonl").read_text(encoding="utf-8"))
        finally:
            import shutil
            shutil.rmtree(private, ignore_errors=True)

    def test_loopback_http_protocols_cli_resume_and_redirect_are_network_safe(self):
        received = []
        generated = {"task": "已有设计解释", "messages": [
            {"role": "user", "content": "解释示例电路。"},
            {"role": "assistant", "content": "根据自编示例资料。"}],
            "evidence": [{"claim": "依据自编示例资料。", "type": "SOURCE_FACT", "source_ids": ["doc-a"]}],
            "expected_points": ["基于可见资料回答"], "hard_errors": []}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = json.loads(raw.decode("utf-8")) if raw else {}
                received.append((self.path, dict(self.headers), body))
                if self.path == "/redirect/responses":
                    self.send_response(302)
                    self.send_header("Location", "/capture")
                    self.end_headers()
                    return
                if self.path == "/v1/responses":
                    payload = {"id": "req-loopback", "status": "completed", "model": "demo-model",
                               "output": [{"type": "message", "role": "assistant", "content": [
                                   {"type": "output_text", "text": json.dumps(generated, ensure_ascii=False)}]}]}
                elif self.path == "/v1/chat/completions":
                    payload = {"id": "req-loopback", "model": "demo-model", "choices": [{
                        "message": {"role": "assistant", "content": "loopback ok"}, "finish_reason": "stop"}]}
                elif self.path == "/v1/messages":
                    payload = {"id": "req-loopback", "model": "demo-model", "content": [
                        {"type": "text", "text": "loopback ok"}], "stop_reason": "end_turn"}
                else:
                    self.send_error(404)
                    return
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}/v1"
        try:
            for provider, api, path in (("openai", "responses", "/v1/responses"),
                                        ("openai", "chat_completions", "/v1/chat/completions"),
                                        ("anthropic", "responses", "/v1/messages")):
                cfg = TeacherConfig(provider, base, FAKE_KEY, "demo-model", api, 128, 2, 0)
                result = call_teacher(cfg, "system", "question")
                self.assertEqual(result["text"], json.dumps(generated, ensure_ascii=False) if path.endswith("responses") else "loopback ok")
                seen_path, headers, body = received[-1]
                self.assertEqual(seen_path, path)
                normalized_headers = {key.casefold(): value for key, value in headers.items()}
                if provider == "anthropic":
                    self.assertEqual(normalized_headers.get("x-api-key"), FAKE_KEY)
                    self.assertEqual(body["system"], "system")
                else:
                    self.assertEqual(normalized_headers.get("authorization"), f"Bearer {FAKE_KEY}")
                    self.assertEqual(body["model"], "demo-model")

            redirect_config = TeacherConfig("openai", f"http://127.0.0.1:{server.server_port}/redirect/responses",
                                            FAKE_KEY, "demo-model", "responses", 128, 1, 0)
            before = len(received)
            with self.assertRaises(TeacherError):
                call_teacher(redirect_config, "system", "question")
            self.assertEqual(len(received), before + 1)
            self.assertEqual(received[-1][0], "/redirect/responses")
            self.assertNotIn("/capture", [request[0] for request in received])

            root = Path(__file__).resolve().parents[1]
            private = root / "本地资料" / "数据准备" / "unittest-http"
            private.mkdir(parents=True, exist_ok=True)
            source, envfile, output = private / "input.jsonl", private / "fake.env", private / "out"
            source.write_text(json.dumps(sample_row(), ensure_ascii=False), encoding="utf-8")
            envfile.write_text(f"TEACHER_PROVIDER=openai\nTEACHER_OPENAI_API=responses\nTEACHER_BASE_URL={base}\n"
                               f"TEACHER_API_KEY={FAKE_KEY}\nTEACHER_MODEL_ID=demo-model\nTEACHER_MAX_RETRIES=0\n", encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if not k.startswith("TEACHER_")}
            env["PYTHONUTF8"] = "1"
            command = [sys.executable, str(root / "scripts" / "prepare_dataset.py"), "--input", str(source),
                       "--env-file", str(envfile), "--output-dir", str(output)]
            try:
                import subprocess
                first = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env)
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                request_total = len(received)
                second = subprocess.run(command + ["--resume"], capture_output=True, text=True, encoding="utf-8", env=env)
                self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
                self.assertEqual(len(received), request_total)
                all_files = "\n".join(p.read_text(encoding="utf-8") for p in output.rglob("*") if p.is_file())
                self.assertNotIn(FAKE_KEY, all_files)
            finally:
                import shutil
                shutil.rmtree(private, ignore_errors=True)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_offline_response_creates_pending_candidate_and_resume_skips_it(self):
        root = Path(__file__).resolve().parents[1]
        private = root / "本地资料" / "数据准备" / "unittest-offline"
        private.mkdir(parents=True, exist_ok=True)
        source, envfile = private / "input.jsonl", private / "empty.env"
        out, prompts, responses = private / "output", private / "prompts", private / "responses"
        responses.mkdir()
        source.write_text(json.dumps(sample_row(), ensure_ascii=False), encoding="utf-8")
        envfile.write_text(f"TEACHER_API_KEY={FAKE_KEY}\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("TEACHER_")}
        env["PYTHONUTF8"] = "1"
        script = str(root / "scripts" / "prepare_dataset.py")

        def run(*args):
            return subprocess.run([sys.executable, script, "--input", str(source), "--env-file", str(envfile),
                                   "--output-dir", str(out), *args], capture_output=True,
                                  text=True, encoding="utf-8", env=env)

        try:
            import subprocess
            exported = run("--export-prompts", "--output-dir", str(prompts))
            self.assertEqual(exported.returncode, 0, exported.stderr)
            task_file = next((prompts / "prompts").glob("*.json"))
            jobid = task_file.stem
            result = {"task": "已有设计解释", "messages": [
                {"role": "user", "content": "解释示例电路。根据自编示例资料。"},
                {"role": "assistant", "content": "这是示例答案。"}],
                "evidence": [{"claim": "资料为自编示例。", "type": "SOURCE_FACT", "source_ids": ["doc-a"]}],
                "expected_points": ["基于可见资料回答"], "hard_errors": []}
            (responses / f"{jobid}.json").write_text(json.dumps({"text": json.dumps(result, ensure_ascii=False),
                "model": FAKE_KEY, "request_id": FAKE_KEY,
                "usage": {"total_tokens": 4, "private_field": FAKE_KEY}}, ensure_ascii=False), encoding="utf-8")
            created = run("--response-dir", str(responses))
            failure_text = (out / "失败记录.jsonl").read_text(encoding="utf-8") if (out / "失败记录.jsonl").exists() else ""
            self.assertEqual(created.returncode, 0, created.stderr + created.stdout + failure_text)
            candidate = json.loads((out / "训练候选.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(candidate["review"]["status"], "待审")
            self.assertFalse(candidate["training_eligible"])
            serialized = "\n".join(p.read_text(encoding="utf-8") for p in out.rglob("*") if p.is_file())
            self.assertNotIn(FAKE_KEY, serialized)
            resumed = run("--response-dir", str(responses), "--resume")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(json.loads(resumed.stdout)["new_successes"], 0)

            changed = sample_row(split="validation", group="moved-group")
            source.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            group_conflict = run("--response-dir", str(responses), "--resume")
            self.assertNotEqual(group_conflict.returncode, 0)

            changed = sample_row(split="validation")
            source.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            split_conflict = run("--response-dir", str(responses), "--resume")
            self.assertNotEqual(split_conflict.returncode, 0)

            changed = sample_row()
            changed["sources"][0]["content"] = "同一来源 ID 的修订内容。"
            source.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            revised = run("--response-dir", str(responses), "--resume")
            self.assertNotEqual(revised.returncode, 0)
            outputs = [json.loads(line) for line in (out / "训练候选.jsonl").read_text(encoding="utf-8").splitlines()]
            ids = [item["sample_id"] for item in outputs]
            self.assertEqual(len(ids), len(set(ids)))
        finally:
            import shutil
            shutil.rmtree(private, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
