"""账户余额模式的离线安全回归测试。"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_demo_data.py"
spec = importlib.util.spec_from_file_location("build_demo_data_balance_test", SCRIPT)
demo = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(demo)


def _inputs(root: Path):
    row = {
        "sample_id": "sample-1", "source_id": "source-1", "group_id": "group-1",
        "split": "train", "task": "给出方案与取舍", "sources": [],
        "title_original": "电阻选择", "question_original": "如何选择？",
        "answer_reference_original": "先核对条件。", "topic": "i2c",
    }
    input_path = root / "input.jsonl"
    approval_path = root / "approval.json"
    input_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    approval_path.write_text(json.dumps({"reviewer": "main", "status": "approved_for_teacher_generation", "source_ids": [row["source_id"]]}), encoding="utf-8")
    config = SimpleNamespace(provider="test", base_url="https://api.example/v1", model_id="model",
        openai_api="responses", api_key="unused", max_retries=3, max_output_tokens=128)
    api = ModuleType("teacher_api")
    api.TeacherError = type("TeacherError", (Exception,), {})
    api.load_config = lambda path, **kwargs: config
    api.load_profile_order = lambda path, case: ("backup", "primary") if case == "data" else ("primary",)
    args = SimpleNamespace(input=input_path, approval_file=approval_path, env_file=root / ".env",
        outdir=root / "本地资料" / "out", limit=1, dry_run=False, resume=False, use_account_balance=True,
        review_notes=None)
    _write_eval_lists(root)
    return args, config, api


def _add_second_source(args):
    rows=[json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    row={**rows[0],"sample_id":"sample-2","source_id":"source-2","group_id":"group-2"}
    rows.append(row)
    args.input.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    approval=json.loads(args.approval_file.read_text(encoding="utf-8"))
    approval["source_ids"].append("source-2")
    args.approval_file.write_text(json.dumps(approval),encoding="utf-8")


def _reply(source_id):
    return {"text":json.dumps({"messages":[{"role":"user","content":"如何选择？"},{"role":"assistant","content":"先核对具体条件与资料。"}],"used_source_ids":[source_id]},ensure_ascii=False),"model":"offline","usage":{"output_tokens":8}}


def _write_eval_lists(root: Path, dev_rows=None, final_rows=None):
    eval_dir = root / "本地资料" / "正式实验" / "评测"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("开发集-冻结-v1.jsonl", dev_rows or []), ("终测集-冻结-v1.jsonl", final_rows or [])):
        (eval_dir / name).write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")


class AccountBalanceSafetyTests(unittest.TestCase):
    def test_same_batch_duplicate_source_or_group_rejected_before_request(self):
        for key in ("source_id", "group_id"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                args, _, _ = _inputs(Path(temp))
                _add_second_source(args)
                rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
                rows[1][key] = rows[0][key]
                args.input.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
                with self.assertRaisesRegex(demo.BuildError, "重复.*未发送 API 请求"):
                    demo._account_targets(args)

    def test_historical_train_reviewed_sourcepack_and_candidate_duplicates_rejected(self):
        for location, duplicate_key in (("training", "sample_id"), ("approved", "source_id"),
                                        ("sourcepack", "group_id"), ("candidate", "group_id")):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args, _, _ = _inputs(root)
                _write_eval_lists(root)
                history = {"sample_id":"old-sample","source_id":"old-source","group_id":"old-group",
                           "split":"train", "sources":[]}
                if location == "training":
                    path = root/"本地资料"/"正式实验"/"训练"/"训练集候选-部分-v1.jsonl"
                elif location == "approved":
                    path = root/"本地资料"/"正式实验"/"训练"/"批次已审-v1.jsonl"
                elif location == "sourcepack":
                    path = root/"本地资料"/"正式实验"/"批次"/"old"/"来源包.jsonl"
                else:
                    path = root/"本地资料"/"正式实验"/"批次"/"old"/"生成结果"/"train_candidates.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                current = json.loads(args.input.read_text(encoding="utf-8"))
                history[duplicate_key] = current[duplicate_key]
                path.write_text(json.dumps(history, ensure_ascii=False)+"\n", encoding="utf-8")
                args.dry_run = True
                with patch.object(demo,"ROOT",root):
                    with self.assertRaisesRegex(demo.BuildError,"重复"):
                        demo.generate(args)

    def test_allow_source_reuse_skips_training_history_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args,_,_=_inputs(root); _write_eval_lists(root)
            row=json.loads(args.input.read_text(encoding="utf-8"))
            row["sample_id"]="old-sample"
            history=root/"本地资料"/"正式实验"/"训练"/"训练集候选-部分-v1.jsonl"
            history.parent.mkdir(parents=True)
            history.write_text(json.dumps(row,ensure_ascii=False)+"\n",encoding="utf-8")
            args.dry_run=True; args.allow_source_reuse=True
            with patch.object(demo,"ROOT",root):
                self.assertEqual(demo.generate(args),0)

    def test_allow_source_reuse_never_allows_sample_id_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args,_,_=_inputs(root); _write_eval_lists(root)
            current=json.loads(args.input.read_text(encoding="utf-8"))
            history={**current,"source_id":"older-source","group_id":"older-group"}
            path=root/"本地资料"/"正式实验"/"训练"/"训练集候选-部分-v1.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(history,ensure_ascii=False)+"\n",encoding="utf-8")
            args.dry_run=True; args.allow_source_reuse=True
            with patch.object(demo,"ROOT",root):
                with self.assertRaisesRegex(demo.BuildError,"sample_id.*重复"):
                    demo.generate(args)

    def test_live_account_reuse_rejected_before_wallet_post_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args,_,api=_inputs(root); _write_eval_lists(root)
            row=json.loads(args.input.read_text(encoding="utf-8"))
            history=root/"本地资料"/"正式实验"/"训练"/"训练集候选-部分-v1.jsonl"
            history.parent.mkdir(parents=True,exist_ok=True)
            history.write_text(json.dumps(row,ensure_ascii=False)+"\n",encoding="utf-8")
            args.dry_run=False
            api.call_teacher=lambda *a: self.fail("重复来源不得 POST")
            private=root/"本地资料"
            with patch.object(demo,"ROOT",root),patch.dict(sys.modules,{"teacher_api":api}), \
                    patch.object(demo,"_wallet",side_effect=AssertionError("重复来源不得查询余额")):
                with self.assertRaisesRegex(demo.BuildError,"重复"):
                    demo.generate(args)
            self.assertFalse((private/".formal_teacher_api.lock").exists())

    def test_frozen_eval_id_and_dataset_row_match_always_rejected_without_text_leak(self):
        for mode in ("source_id", "dataset_row"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); args,_,_=_inputs(root)
                row=json.loads(args.input.read_text(encoding="utf-8"))
                eval_row={"source_id":"frozen-other","group_id":"frozen-group","project_id":"frozen-project",
                          "input_messages":[{"role":"user","content":"SECRET_QUESTION_SENTINEL"}],
                          "scoring":{"SECRET_SCORING_SENTINEL":1}}
                if mode=="source_id":
                    eval_row["source_id"]=row["source_id"]
                else:
                    row["sources"]=[{"dataset":"dataset-X","revision":"rev-X","row_index":17}]
                    args.input.write_text(json.dumps(row,ensure_ascii=False)+"\n",encoding="utf-8")
                    eval_row["sources"]=[{"dataset":"dataset-X","revision":"rev-X","row_index":17}]
                _write_eval_lists(root,dev_rows=[eval_row])
                args.dry_run=True; args.allow_source_reuse=True
                with patch.object(demo,"ROOT",root):
                    with self.assertRaises(demo.BuildError) as caught:
                        demo.generate(args)
                self.assertNotIn("SECRET_QUESTION_SENTINEL",str(caught.exception))
                self.assertNotIn("SECRET_SCORING_SENTINEL",str(caught.exception))

    def test_missing_frozen_lists_fail_closed_and_dry_run_never_checks_wallet_or_posts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args,_,api=_inputs(root); args.dry_run=True
            eval_dir=root/"本地资料"/"正式实验"/"评测"
            for path in eval_dir.glob("*.jsonl"):
                path.unlink()
            api.call_teacher=lambda *a: self.fail("dry-run must not POST")
            with patch.object(demo,"ROOT",root),patch.object(demo,"_wallet",side_effect=AssertionError("dry-run must not query wallet")):
                with self.assertRaisesRegex(demo.BuildError,"清单缺失"):
                    demo.generate(args)
            _write_eval_lists(root)
            with patch.object(demo,"ROOT",root),patch.object(demo,"_wallet",side_effect=AssertionError("dry-run must not query wallet")):
                self.assertEqual(demo.generate(args),0)

    def test_zero_balance_never_calls_teacher(self):
        with tempfile.TemporaryDirectory() as temp:
            args, config, api = _inputs(Path(temp))
            api.call_teacher = lambda *a: self.fail("零余额时不得调用教师")
            with patch.object(demo, "ROOT", Path(temp)), patch.dict(sys.modules, {"teacher_api": api}), patch.object(demo, "_wallet", return_value=0), \
                    patch.object(demo, "_ensure_tokenizer"):
                with self.assertRaisesRegex(demo.BuildError, "余额为零"):
                    demo.generate(args)
            ledger=json.loads((args.outdir/".teacher_account_balance_run.json").read_text(encoding="utf-8"))
            self.assertEqual((ledger["status"],ledger["total"],ledger["generated"],ledger["pending_generation"]),
                             ("balance_zero",1,0,1))
            self.assertEqual(config.max_retries,0)

    def test_timeout_marks_request_unknown_and_resume_never_reposts(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, api = _inputs(Path(temp))
            calls = []
            def timeout(*a):
                calls.append("POST")
                # teacher_api 将网络超时包装为无 HTTP 状态、无分类的 TeacherError。
                raise api.TeacherError("教师接口请求失败；连接超时")
            api.call_teacher = timeout
            with patch.object(demo, "ROOT", Path(temp)), patch.dict(sys.modules, {"teacher_api": api}), patch.object(demo, "_wallet", return_value=5), \
                    patch.object(demo, "_ensure_tokenizer"), patch.object(demo, "_token_count", return_value=20):
                with self.assertRaises(demo.BuildError):
                    demo.generate(args)
                ledger = json.loads((args.outdir / ".teacher_account_balance_run.json").read_text(encoding="utf-8"))
                self.assertEqual(ledger.get("status"), "request_unknown")
                args.resume = True
                with self.assertRaises(demo.BuildError):
                    demo.generate(args)
            self.assertEqual(calls, ["POST"])

    def test_resume_generates_two_batches_and_accumulates_ledger_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, api = _inputs(Path(temp)); _add_second_source(args)
            calls=[]
            api.call_teacher=lambda config,system,prompt: (calls.append("POST") or _reply("source-"+str(len(calls))))
            # 余额查询次数随每批的前置检查、请求前检查和请求后检查变化，mock 持续返回有效余额。
            with patch.object(demo,"ROOT",Path(temp)),patch.dict(sys.modules,{"teacher_api":api}),patch.object(demo,"_wallet",return_value=10),patch.object(demo,"_ensure_tokenizer"),patch.object(demo,"_token_count",return_value=20):
                self.assertEqual(demo.generate(args),0)
                first=json.loads((args.outdir/".teacher_account_balance_run.json").read_text(encoding="utf-8"))
                args.resume=True
                self.assertEqual(demo.generate(args),0)
            ledger=json.loads((args.outdir/".teacher_account_balance_run.json").read_text(encoding="utf-8"))
            self.assertEqual(calls,["POST","POST"])
            self.assertEqual(ledger["starting_balance"],first["starting_balance"])
            self.assertEqual((ledger["total"],ledger["generated"],ledger["pending_generation"]),(2,2,0))

    def test_resume_reuses_saved_reply_when_candidate_conversion_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, api = _inputs(Path(temp)); calls=[]
            api.call_teacher=lambda *a: (calls.append("POST") or _reply("source-1"))
            original=demo._candidate_from_reply
            def fail_once(*a,**kw):
                if not getattr(fail_once,"failed",False):
                    fail_once.failed=True
                    raise demo.BuildError("simulated candidate conversion failure")
                return original(*a,**kw)
            with patch.object(demo,"ROOT",Path(temp)),patch.dict(sys.modules,{"teacher_api":api}),patch.object(demo,"_wallet",return_value=5),patch.object(demo,"_ensure_tokenizer"),patch.object(demo,"_token_count",return_value=20),patch.object(demo,"_candidate_from_reply",side_effect=fail_once):
                with self.assertRaisesRegex(demo.BuildError,"simulated candidate"):
                    demo.generate(args)
                raw_path=args.outdir/"teacher_raw"/"sample-1.json"
                self.assertTrue(raw_path.is_file())
                args.resume=True
                self.assertEqual(demo.generate(args),0)
            self.assertEqual(calls,["POST"])
            self.assertTrue((args.outdir/"train_candidates.jsonl").is_file())

    def test_server_error_marks_request_unknown_and_resume_never_reposts(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, api = _inputs(Path(temp))
            calls = []
            def server_error(*a):
                calls.append("POST")
                error = api.TeacherError("gateway error")
                error.http_status = 500
                error.category = "unknown"
                raise error
            api.call_teacher = server_error
            with patch.object(demo, "ROOT", Path(temp)), patch.dict(sys.modules, {"teacher_api": api}), patch.object(demo, "_wallet", return_value=5), \
                    patch.object(demo, "_ensure_tokenizer"), patch.object(demo, "_token_count", return_value=20):
                with self.assertRaises(demo.BuildError):
                    demo.generate(args)
                ledger = json.loads((args.outdir / ".teacher_account_balance_run.json").read_text(encoding="utf-8"))
                self.assertEqual(ledger.get("status"), "request_unknown")
                args.resume = True
                with self.assertRaises(demo.BuildError):
                    demo.generate(args)
            self.assertEqual(calls, ["POST"])

    def test_rate_limit_and_auth_are_not_reported_as_empty_balance(self):
        for status,category in [(401,"auth"),(403,"auth"),(429,"rate_limit")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                args,_,api=_inputs(Path(temp))
                error=api.TeacherError("safe failure")
                error.http_status=status;error.category=category
                api.call_teacher=lambda *a: (_ for _ in ()).throw(error)
                with patch.object(demo,"ROOT",Path(temp)),patch.dict(sys.modules,{"teacher_api":api}), \
                        patch.object(demo,"_wallet",return_value=5),patch.object(demo,"_ensure_tokenizer"):
                    with self.assertRaises(demo.BuildError):demo.generate(args)
                ledger=json.loads((args.outdir/".teacher_account_balance_run.json").read_text(encoding="utf-8"))
                self.assertEqual((ledger['status'],ledger['last_error_category']),('request_failed',category))
                self.assertIsNone(ledger['last_error_reason'])

    def test_explicit_insufficient_balance_keeps_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            args,_,api=_inputs(Path(temp))
            error=api.TeacherError("safe failure")
            error.http_status=402;error.category="quota_billing";error.reason="insufficient_balance"
            api.call_teacher=lambda *a: (_ for _ in ()).throw(error)
            with patch.object(demo,"ROOT",Path(temp)),patch.dict(sys.modules,{"teacher_api":api}), \
                    patch.object(demo,"_wallet",return_value=.1),patch.object(demo,"_ensure_tokenizer"):
                with self.assertRaises(demo.BuildError):demo.generate(args)
            ledger=json.loads((args.outdir/".teacher_account_balance_run.json").read_text(encoding="utf-8"))
            self.assertEqual((ledger['status'],ledger['total'],ledger['generated'],ledger['pending_generation']),
                             ('balance_insufficient',1,0,1))

    def test_wallet_failure_after_success_keeps_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            args,_,api=_inputs(Path(temp));calls=[]
            api.call_teacher=lambda *a: (calls.append('POST') or _reply('source-1'))
            with patch.object(demo,"ROOT",Path(temp)),patch.dict(sys.modules,{"teacher_api":api}), \
                    patch.object(demo,"_wallet",side_effect=[5,5,demo.BuildError('wallet unavailable')]), \
                    patch.object(demo,"_ensure_tokenizer"),patch.object(demo,"_token_count",return_value=20):
                with self.assertRaises(demo.BuildError):demo.generate(args)
            ledger=json.loads((args.outdir/".teacher_account_balance_run.json").read_text(encoding="utf-8"))
            self.assertEqual((ledger['status'],ledger['generated'],ledger['pending_generation']),('balance_unknown_saved',1,0))
            self.assertTrue((args.outdir/'train_candidates.jsonl').is_file())
            self.assertEqual(calls,['POST'])

    def test_unavailable_balance_saves_safe_reason_and_never_calls_teacher(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, api = _inputs(Path(temp))
            api.call_teacher = lambda *a: self.fail('余额不可确认时不能调用')
            with patch.object(demo,'ROOT',Path(temp)),patch.dict(sys.modules,{'teacher_api':api}), \
                    patch.object(demo,'_wallet',side_effect=demo.BuildError('余额接口安全HTTP错误：503')), \
                    patch.object(demo,'_ensure_tokenizer'):
                with self.assertRaisesRegex(demo.BuildError,'余额无法确认'): demo.generate(args)
            ledger=json.loads((args.outdir/'.teacher_account_balance_run.json').read_text(encoding='utf-8'))
            self.assertEqual((ledger['status'],ledger['balance_error_category'],ledger['balance_error_http_status']),
                             ('balance_unknown','http_error',503))
            self.assertEqual((ledger['generated'],ledger['pending_generation']),(0,1))

    def test_balance_endpoint_disabled_skips_wallet_and_generates(self):
        with tempfile.TemporaryDirectory() as temp:
            args,_,api=_inputs(Path(temp));calls=[]
            args.balance_endpoint_disabled=True
            api.call_teacher=lambda *a:(calls.append('POST') or _reply('source-1'))
            with patch.object(demo,'ROOT',Path(temp)),patch.dict(sys.modules,{'teacher_api':api}), \
                    patch.object(demo,'_wallet',side_effect=AssertionError('不应请求余额接口')), \
                    patch.object(demo,'_ensure_tokenizer'),patch.object(demo,'_token_count',return_value=20):
                self.assertEqual(demo.generate(args),0)
            ledger=json.loads((args.outdir/'.teacher_account_balance_run.json').read_text(encoding='utf-8'))
            self.assertEqual((ledger['status'],ledger['generated'],ledger['pending_generation']),
                             ('generation_complete',1,0))
            self.assertEqual(ledger['balance_error_category'],'balance_endpoint_disabled')
            self.assertEqual(calls,['POST'])

    def test_balance_endpoint_disabled_still_stops_on_quota_error(self):
        with tempfile.TemporaryDirectory() as temp:
            args,_,api=_inputs(Path(temp));calls=[]
            args.balance_endpoint_disabled=True
            error=api.TeacherError('额度受限')
            error.http_status=402;error.category='quota_billing'
            error.reason='insufficient_balance';error.retry_after_seconds=None
            def fail(*a):
                calls.append('POST')
                raise error
            api.call_teacher=fail
            with patch.object(demo,'ROOT',Path(temp)),patch.dict(sys.modules,{'teacher_api':api}), \
                    patch.object(demo,'_wallet',side_effect=AssertionError('不应请求余额接口')), \
                    patch.object(demo,'_ensure_tokenizer'),patch.object(demo,'_token_count',return_value=20):
                with self.assertRaisesRegex(demo.BuildError,'quota_billing'):
                    demo.generate(args)
            ledger=json.loads((args.outdir/'.teacher_account_balance_run.json').read_text(encoding='utf-8'))
            self.assertEqual((ledger['status'],ledger['generated']),('balance_insufficient',0))
            self.assertEqual(calls,['POST'])

    def test_saved_reply_is_recovered_after_interruption_before_ledger_update(self):
        with tempfile.TemporaryDirectory() as temp:
            args,_,api=_inputs(Path(temp));calls=[]
            api.call_teacher=lambda *a: (calls.append('POST') or _reply('source-1'))
            real_write=demo._atomic_json
            def interrupted(path,value):
                if path.name=='.teacher_account_balance_run.json' and value.get('status')=='response_saved':
                    raise RuntimeError('simulated interruption')
                return real_write(path,value)
            with patch.object(demo,"ROOT",Path(temp)),patch.dict(sys.modules,{"teacher_api":api}), \
                    patch.object(demo,"_wallet",return_value=5),patch.object(demo,"_ensure_tokenizer"), \
                    patch.object(demo,"_token_count",return_value=20):
                with patch.object(demo,'_atomic_json',side_effect=interrupted):
                    with self.assertRaises(RuntimeError):demo.generate(args)
                ledger=json.loads((args.outdir/'.teacher_account_balance_run.json').read_text(encoding='utf-8'))
                self.assertEqual(ledger['status'],'request_unknown')
                self.assertTrue((args.outdir/'teacher_raw'/'sample-1.json').is_file())
                args.resume=True
                self.assertEqual(demo.generate(args),0)
            self.assertEqual(calls,['POST'])

    def test_existing_lock_prevents_concurrent_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args,_,api=_inputs(root);private=root/'本地资料';private.mkdir(exist_ok=True)
            lock=private/'.formal_teacher_api.lock';lock.write_text('{"pid":1,"lock_id":"other"}',encoding='utf-8')
            api.call_teacher=lambda *a: self.fail('已有锁时不应发请求')
            with patch.object(demo,'ROOT',root),patch.dict(sys.modules,{'teacher_api':api}):
                with self.assertRaisesRegex(demo.BuildError,'已有账户余额任务'):demo.generate(args)
            self.assertTrue(lock.is_file())

    def test_concurrency_slots_allow_bounded_parallel_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args,_,api=_inputs(root);private=root/'本地资料';private.mkdir(exist_ok=True)
            (private/'.formal_teacher_api.lock.1').write_text('{"pid":1}',encoding='utf-8')
            api.call_teacher=lambda *a:_reply('source-1')
            with patch.dict(os.environ,{'FORMAL_TEACHER_SLOTS':'2'}),patch.object(demo,'ROOT',root), \
                    patch.dict(sys.modules,{'teacher_api':api}),patch.object(demo,'_wallet',return_value=5), \
                    patch.object(demo,'_ensure_tokenizer'),patch.object(demo,'_token_count',return_value=20):
                self.assertEqual(demo.generate(args),0)
            self.assertFalse((private/'.formal_teacher_api.lock.2').exists())
            self.assertTrue((private/'.formal_teacher_api.lock.1').is_file())

    def test_formal_multiturn_limit_keeps_complete_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            args,_,_=_inputs(Path(temp))
            row=json.loads(args.input.read_text(encoding='utf-8'))
            row['task']='多轮需求修订'
            messages=[]
            for i in range(5):
                messages.extend([{'role':'user','content':f'补充条件 {i}'},
                                 {'role':'assistant','content':f'根据条件 {i} 更新建议'}])
            reply=json.dumps({'messages':messages,'used_source_ids':[row['source_id']]},ensure_ascii=False)
            with patch.object(demo,'_token_count',return_value=2048):
                candidate=demo._candidate_from_reply(row,reply,'offline',None,max_messages=12,max_tokens=4096)
                self.assertEqual(candidate['messages'],messages)
                self.assertEqual(candidate['task'],'多轮需求修订')
                with self.assertRaisesRegex(demo.BuildError,'超过1024'):
                    demo._candidate_from_reply(row,reply,'offline',None,max_messages=12,max_tokens=1024)
                with self.assertRaisesRegex(demo.BuildError,'消息数量'):
                    demo._candidate_from_reply(row,reply,'offline',None)

    def test_final_source_and_cross_split_group_are_rejected_before_call(self):
        for kind in ('final', 'group'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                args, _, api = _inputs(Path(temp))
                _add_second_source(args)
                rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines()]
                if kind == 'final':
                    rows[1]['split'] = 'test'
                else:
                    rows[1]['split'] = 'validation'
                    rows[1]['group_id'] = rows[0]['group_id']
                args.input.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf-8')
                api.call_teacher = lambda *a: self.fail('不合规来源不得发送')
                with patch.object(demo, 'ROOT', Path(temp)), patch.dict(sys.modules, {'teacher_api': api}), \
                        patch.object(demo, '_wallet', side_effect=AssertionError('不应读取余额')):
                    with self.assertRaises(demo.BuildError): demo.generate(args)

    def test_changed_manifest_cannot_resume_or_send(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, api = _inputs(Path(temp)); calls = []
            api.call_teacher = lambda *a: (calls.append('POST') or _reply('source-1'))
            with patch.object(demo, 'ROOT', Path(temp)), patch.dict(sys.modules, {'teacher_api': api}), \
                    patch.object(demo, '_wallet', return_value=5), patch.object(demo, '_ensure_tokenizer'), \
                    patch.object(demo, '_token_count', return_value=20):
                demo.generate(args)
                row = json.loads(args.input.read_text(encoding='utf-8'))
                row['question_original'] = '来源发生变更'
                args.input.write_text(json.dumps(row, ensure_ascii=False)+'\n', encoding='utf-8')
                args.resume = True
                with self.assertRaisesRegex(demo.BuildError, '不一致'): demo.generate(args)
            self.assertEqual(calls, ['POST'])

    def test_account_outputs_cannot_escape_private_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); args, _, api = _inputs(root)
            args.outdir = root / 'public_output'
            api.call_teacher = lambda *a: self.fail('公共目录不得发送')
            with patch.object(demo, 'ROOT', root), patch.dict(sys.modules, {'teacher_api': api}):
                with self.assertRaisesRegex(demo.BuildError, '本地资料'): demo.generate(args)
            self.assertFalse(args.outdir.exists())

    def test_multiturn_prompt_explicitly_requests_two_turns(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _ = _inputs(Path(temp))
            row = json.loads(args.input.read_text(encoding='utf-8'))
            row['task'] = '多轮需求修订'
            self.assertIn('至少两轮对话', demo._teacher_prompt(row, formal=True, max_messages=12))

    def test_export_keeps_task_and_review_evidence_outside_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _ = _inputs(root)
            row = json.loads(args.input.read_text(encoding='utf-8'))
            with patch.object(demo, '_token_count', return_value=20):
                candidate = demo._candidate_from_reply(row, _reply('source-1')['text'], 'offline', None)
            candidate['review'] = {'status': 'approved', 'evidence': {'UNCONFIRMED': '具体参数待查'}}
            source = root / 'candidate.jsonl'; output = root / 'approved.jsonl'
            source.write_text(json.dumps(candidate, ensure_ascii=False)+'\n', encoding='utf-8')
            export_args = SimpleNamespace(input=source, output=output, approved_by='main', max_message_count=12)
            demo.export_approved(export_args)
            exported = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(exported['task'], row['task'])
            self.assertEqual(exported['evidence'], candidate['review']['evidence'])
            self.assertEqual(exported['teacher'], candidate['teacher'])
            self.assertEqual(exported['messages'], candidate['messages'])

    def test_general_dialogue_uses_original_conversation_and_own_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, api = _inputs(Path(temp)); args.general_dialogue = True
            row = json.loads(args.input.read_text(encoding='utf-8'))
            row['scenario'] = [{'role':'user','content':'请帮我规划会议'},
                               {'role':'assistant','content':'先确认会议主题。'}]
            args.input.write_text(json.dumps(row, ensure_ascii=False)+'\n', encoding='utf-8')
            calls = []
            def call(config, system, prompt):
                calls.append((system,prompt))
                return _reply('source-1')
            api.call_teacher = call
            with patch.object(demo,'ROOT',Path(temp)),patch.dict(sys.modules,{'teacher_api':api}), \
                    patch.object(demo,'_wallet',return_value=5),patch.object(demo,'_ensure_tokenizer'), \
                    patch.object(demo,'_token_count',return_value=20):
                demo.generate(args)
                args.general_dialogue = False; args.resume = True
                with self.assertRaisesRegex(demo.BuildError,'不一致'): demo.generate(args)
            self.assertEqual(len(calls),1)
            self.assertIn('日常交流',calls[0][0])
            self.assertIn('请帮我规划会议',calls[0][1])


if __name__ == "__main__":
    unittest.main()
