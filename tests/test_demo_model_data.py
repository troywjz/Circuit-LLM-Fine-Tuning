"""使用当前本机 Qwen tokenizer 检查 Demo 的真实模板与 assistant-only 标签。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from demo_data_utils import DemoDataError, read_jsonl, tokenize_assistant_only
from demo_infer import _append_turn, _load_existing_output, _read_cases, _run_fingerprint
import demo_infer
from qwen35_common import DEFAULT_MODEL_PATH, effective_chat_stop_ids


class DemoModelDataTests(unittest.TestCase):
    _tokenizer = None

    def get_tokenizer(self):
        if self.__class__._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.__class__._tokenizer = AutoTokenizer.from_pretrained(
                    str(DEFAULT_MODEL_PATH), local_files_only=True, trust_remote_code=False)
            except (ImportError, OSError) as exc:
                self.skipTest(f"本地tokenizer不可用：{exc}")
        return self.__class__._tokenizer

    def test_effective_chat_stop_ids_adds_tokenizer_and_chat_eos(self):
        class Tokenizer:
            eos_token_id = 11
            unk_token_id = 0
            def convert_tokens_to_ids(self, token):
                return {"<|im_end|>": 12}.get(token, 0)
            def get_vocab(self):
                return {"<|im_end|>": 12}

        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=10))
        self.assertEqual(effective_chat_stop_ids(model, Tokenizer()), [10, 11, 12])

    def test_effective_chat_stop_ids_deduplicates_complete_eos_set(self):
        class Tokenizer:
            eos_token_id = 12
            unk_token_id = 0
            def convert_tokens_to_ids(self, token):
                return 12
            def get_vocab(self):
                return {"<|im_end|>": 12}

        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=(12, 10, 12)))
        self.assertEqual(effective_chat_stop_ids(model, Tokenizer()), [12, 10])

    def test_effective_chat_stop_ids_rejects_missing_chat_eos_and_bad_ids(self):
        class MissingChatEos:
            eos_token_id = 11
            unk_token_id = 0
            def convert_tokens_to_ids(self, _token):
                return 0
            def get_vocab(self):
                return {}

        class Tokenizer(MissingChatEos):
            def convert_tokens_to_ids(self, _token):
                return 12
            def get_vocab(self):
                return {"<|im_end|>": 12}

        with self.assertRaisesRegex(ValueError, "未识别"):
            effective_chat_stop_ids(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1)), MissingChatEos())
        for bad_id in (True, -1, "12", 1.5):
            with self.subTest(bad_id=bad_id), self.assertRaisesRegex(ValueError, "非负整数"):
                effective_chat_stop_ids(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=bad_id)), Tokenizer())

    def test_fingerprint_changes_with_stop_protocol_or_local_generation_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = root / "cases.jsonl"
            cases.write_text("{}\n", encoding="utf-8")
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            args = (cases, model, None, "Base", 0, 4096, 1024, 3407)
            baseline, _ = _run_fingerprint(*args)
            with patch.object(demo_infer, "CHAT_STOP_POLICY_VERSION", "qwen35-chat-stop-next"):
                self.assertNotEqual(_run_fingerprint(*args)[0], baseline)
            generation_config = model / "generation_config.json"
            generation_config.write_text('{"eos_token_id": [1, 2]}', encoding="utf-8")
            configured, _ = _run_fingerprint(*args)
            self.assertNotEqual(configured, baseline)
            generation_config.write_text('{"eos_token_id": [1, 3]}', encoding="utf-8")
            self.assertNotEqual(_run_fingerprint(*args)[0], configured)

    def test_single_turn_masks_user_and_supervises_answer_and_eos(self):
        tokenizer = self.get_tokenizer()
        messages = [{"role": "user", "content": "请解释 1 kΩ。"},
                    {"role": "assistant", "content": "1 kΩ 等于 1000 Ω。"}]
        result = tokenize_assistant_only(tokenizer, messages, 256)
        target = [x for x in result["labels"] if x != -100]
        self.assertEqual(target[-1], tokenizer.eos_token_id)
        self.assertGreater(len(target), 1)
        self.assertEqual(tokenizer.decode(target[:-1]), messages[-1]["content"])
        self.assertEqual(result["token_count"], len(result["input_ids"]))

    def test_multiturn_supervises_each_assistant_turn_including_each_eos(self):
        tokenizer = self.get_tokenizer()
        messages = [{"role": "user", "content": "先问清楚电源。"},
                    {"role": "assistant", "content": "电源电压范围是多少？"},
                    {"role": "user", "content": "3.3 V。"},
                    {"role": "assistant", "content": "收到，按 3.3 V 作为已知条件。"}]
        result = tokenize_assistant_only(tokenizer, messages, 256)
        eos_positions = [i for i, token in enumerate(result["labels"]) if token == tokenizer.eos_token_id]
        self.assertEqual(len(eos_positions), 2)
        supervised = [x for x in result["labels"] if x != -100]
        decoded = tokenizer.decode([x for x in supervised if x != tokenizer.eos_token_id])
        self.assertEqual(decoded, messages[1]["content"] + messages[3]["content"])
        self.assertGreater(result["assistant_label_tokens"], len(eos_positions))

    def test_long_sample_is_rejected_without_truncation(self):
        tokenizer = self.get_tokenizer()
        messages = [{"role": "user", "content": "x" * 400},
                    {"role": "assistant", "content": "答复。"}]
        with self.assertRaisesRegex(DemoDataError, "未截断"):
            tokenize_assistant_only(tokenizer, messages, 32)

    def test_duplicate_ids_and_unapproved_rows_are_rejected(self):
        row = {"sample_id": "demo-a", "source_id": "src-a", "group_id": "grp-a",
               "split": "train", "approved_for_demo": True,
               "messages": [{"role": "user", "content": "问题"},
                            {"role": "assistant", "content": "回答"}],
               "review": {"status": "approved"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n" +
                            json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DemoDataError, "重复"):
                read_jsonl(path)
            row["sample_id"] = "demo-b"
            row["approved_for_demo"] = False
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DemoDataError, "approved_for_demo"):
                read_jsonl(path)

    def test_inference_case_reader_drops_gold_and_rubric(self):
        row = {"case_id": "case-1", "kind": "domain", "split": "dev",
               "input_messages": [{"role": "user", "content": "问题"}],
               "gold": "不可进入模型的参考答案", "rubric": {"answer": "私有评分信息"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            case = _read_cases(path, None)[0]
            self.assertNotIn("gold", case)
            self.assertNotIn("rubric", case)

    def test_inference_fingerprint_changes_when_checkpoint_weights_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = root / "cases.jsonl"
            cases.write_text("{}\n", encoding="utf-8")
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            adapter = root / "adapter-step-15"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text('{"r":8}', encoding="utf-8")
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"checkpoint A")
            first, _ = _run_fingerprint(cases, model, adapter, "FT", 15, 2048, 384, 3407)
            weights.write_bytes(b"checkpoint B")
            second, _ = _run_fingerprint(cases, model, adapter, "FT", 15, 2048, 384, 3407)
            self.assertNotEqual(first, second)

    def test_resume_rejects_mixed_fingerprint_and_duplicate_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text("not json\n", encoding="utf-8")
            with self.assertRaisesRegex(DemoDataError, "不是有效 JSON"):
                _load_existing_output(path, "new")
            path.write_text(json.dumps({"case_id": "case-a", "run_fingerprint": "old"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DemoDataError, "不同运行指纹"):
                _load_existing_output(path, "new")
            duplicate = {"case_id": "case-a", "run_fingerprint": "same"}
            path.write_text(json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DemoDataError, "重复 case_id"):
                _load_existing_output(path, "same")

    def test_multiturn_history_uses_own_prior_answer(self):
        history = [{"role": "user", "content": "原问题"},
                   {"role": "assistant", "content": "模型自己的回答"}]
        next_history = _append_turn(history, "用户更正", "模型更新后的回答")
        self.assertEqual(next_history[-2:], [{"role": "user", "content": "用户更正"},
                                             {"role": "assistant", "content": "模型更新后的回答"}])
        self.assertNotIn("教师参考答案", json.dumps(next_history, ensure_ascii=False))

    def test_main_sends_user_correction_once_and_saves_the_same_history(self):
        class FakeIds:
            def __init__(self, count, last):
                self.count = count
                self.last = last
            def numel(self):
                return self.count
            def __getitem__(self, key):
                return self.last if key == -1 else 1

        class FakeGenerated:
            def __init__(self, count, last):
                self.count = count
                self.last = last
            def __getitem__(self, key):
                return FakeIds(self.count, self.last)

        class FakeTensor:
            def __init__(self, count):
                self.shape = (1, count)
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __init__(self):
                self.histories = []
                self.generation_index = 0
                self.answers = ["先按一般负载估算。", "已按3.3V约束调整，改用对应电源轨。"]
                self.eos_token_id = 11
                self.unk_token_id = 0
            def convert_tokens_to_ids(self, token):
                return {"<|im_end|>": 12}.get(token, 0)
            def get_vocab(self):
                return {"<|im_end|>": 12}
            def apply_chat_template(self, messages, **_kwargs):
                self.histories.append(json.loads(json.dumps(messages, ensure_ascii=False)))
                return {"input_ids": FakeTensor(len(json.dumps(messages, ensure_ascii=False)))}
            def decode(self, _ids, **_kwargs):
                return self.answers[self.generation_index - 1]

        tokenizer = FakeTokenizer()
        class FakeModel:
            device = "cpu"
            generation_config = SimpleNamespace(eos_token_id=10)
            generation_stop_ids = []
            def eval(self):
                return self
            def generate(self, **kwargs):
                self.generation_stop_ids.append(kwargs["eos_token_id"])
                tokenizer.generation_index += 1
                # 没有 <|im_end|> 停止 ID 时，模拟生成器继续写出 user 角色 token。
                if 12 in kwargs["eos_token_id"]:
                    return FakeGenerated(1, 12)
                return FakeGenerated(2, 90)

        fake_torch = SimpleNamespace(
            manual_seed=lambda _seed: None,
            cuda=SimpleNamespace(manual_seed_all=lambda _seed: None,
                reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
                max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
                is_available=lambda: False),
            inference_mode=lambda: __import__("contextlib").nullcontext())
        fake_unsloth = ModuleType("unsloth")
        class FakeFastLanguageModel:
            @staticmethod
            def for_inference(_model):
                pass
        fake_unsloth.FastLanguageModel = FakeFastLanguageModel

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "base"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            cases_path = root / "cases.jsonl"
            case = {"case_id": "multi-1", "kind": "domain", "split": "dev",
                    "input_messages": [{"role": "system", "content": "按题面回答"},
                                       {"role": "user", "content": "最初没有给电压约束"}],
                    "user_turns": ["更正：输入必须兼容3.3V，请按此更新方案。"],
                    "scoring": {"gold": "不可进入模型"}}
            cases_path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
            output_path = root / "answers.jsonl"
            model = FakeModel()
            with patch.dict(sys.modules, {"unsloth": fake_unsloth, "torch": fake_torch}), \
                 patch.object(demo_infer, "load_unsloth_model",
                              return_value=(FakeFastLanguageModel, model, tokenizer, fake_torch)):
                result = demo_infer.main(["--cases", str(cases_path), "--output", str(output_path),
                    "--model-path", str(model_dir), "--seq-length", "2048", "--max-new-tokens", "1"])
            self.assertEqual(result, 0)
            self.assertEqual(len(tokenizer.histories), 2)
            correction = "更正：输入必须兼容3.3V，请按此更新方案。"
            self.assertEqual(sum(m["content"] == correction for m in tokenizer.histories[1]), 1)
            self.assertEqual(tokenizer.histories[1][-1], {"role": "user", "content": correction})
            saved = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(saved["full_dialogue"], tokenizer.histories[1] + [
                {"role": "assistant", "content": "已按3.3V约束调整，改用对应电源轨。"}])
            self.assertEqual(saved["full_dialogue"].count({"role": "user", "content": correction}), 1)
            self.assertIn("3.3V约束", saved["answers_by_turn"][1])
            expected_stops = [10, 11, 12]
            self.assertEqual(model.generation_stop_ids, [expected_stops, expected_stops])
            self.assertEqual(saved["stop_token_ids"], expected_stops)
            self.assertEqual(saved["stop_policy_version"], demo_infer.CHAT_STOP_POLICY_VERSION)
            self.assertTrue(all(turn["stop_token_ids"] == expected_stops for turn in saved["turns"]))
            self.assertTrue(all(turn["stop_policy_version"] == demo_infer.CHAT_STOP_POLICY_VERSION
                                for turn in saved["turns"]))
            self.assertTrue(all(turn["finish_reason"] == "completed" and not turn["truncated"]
                                for turn in saved["turns"]))
            self.assertTrue(all(turn["output_tokens"] == 1 for turn in saved["turns"]))
            self.assertTrue(all(turn["last_token_id"] == 12 and turn["stopped_on_eos"]
                                for turn in saved["turns"]))


if __name__ == "__main__":
    unittest.main()
