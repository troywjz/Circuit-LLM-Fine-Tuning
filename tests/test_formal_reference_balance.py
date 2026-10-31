"""正式强参考账户模式：离线验证余额、消息隔离和断点续跑。"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_demo_data
import demo_api_reference as reference
from teacher_api import TeacherConfig, TeacherError


class FormalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.private = self.root / "本地资料"
        self.private.mkdir()
        self.cases = self.private / "cases.jsonl"
        self.output = self.private / "reference.jsonl"
        self.state = self.private / "account.json"
        self.config = TeacherConfig("openai", "https://example.invalid/v1", "unit-test-only",
                                    "configured-model", "responses", 8192, 1, 0)
        self.case = {"case_id": "c1", "kind": "general", "split": "test",
                     "input_messages": [{"role": "user", "content": "first question"}],
                     "user_turns": ["revision"], "gold": "PRIVATE GOLD", "rubric": "PRIVATE RUBRIC"}
        self.write_cases([self.case])

    def tearDown(self):
        self.temp.cleanup()

    def write_cases(self, rows):
        self.cases.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in rows), encoding="utf-8")

    def args(self, *extra):
        return ["--cases", str(self.cases), "--output", str(self.output),
                "--budget-state", str(self.state), "--env-file", str(self.private / "missing.env"),
                "--use-account-balance", *extra]

    def test_zero_balance_stops_before_post_and_records_progress(self):
        with patch.object(build_demo_data, "ROOT", self.root), \
             patch("teacher_api.load_config", return_value=self.config), \
             patch.object(reference, "_wallet", return_value=0), \
             patch("teacher_api.call_teacher_messages") as caller:
            self.assertEqual(reference.main(self.args()), 2)
        caller.assert_not_called()
        state = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual((state["status"], state["completed_turns"], state["pending_turns"]), ("balance_zero", 0, 2))

    def test_positive_balance_below_old_reserve_can_run(self):
        self.case["user_turns"] = []; self.write_cases([self.case])
        response = {"text":"answer", "usage":{}, "finish_reason":"end_turn", "reported_model":"actual"}
        with patch.object(build_demo_data, "ROOT", self.root), patch("teacher_api.load_config", return_value=self.config), \
             patch.object(reference, "_wallet", return_value=.1), patch("teacher_api.call_teacher_messages", return_value=response) as caller:
            self.assertEqual(reference.main(self.args()), 0)
        self.assertEqual(caller.call_count, 1)
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8"))["requested_model"], "configured-model")

    def test_timeout_or_server_failure_is_unknown_and_never_reposted(self):
        for status in (None, 500):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); private = root / "本地资料"; private.mkdir()
                cases = private / "cases.jsonl"; output = private / "out.jsonl"; state = private / "state.json"
                cases.write_text(json.dumps(self.case)+"\n", encoding="utf-8")
                args = ["--cases",str(cases),"--output",str(output),"--budget-state",str(state),"--use-account-balance"]
                error = TeacherError("safe failure", http_status=status, category="unknown")
                with patch.object(build_demo_data,"ROOT",root),patch("teacher_api.load_config",return_value=self.config), \
                     patch.object(reference,"_wallet",return_value=5),patch("teacher_api.call_teacher_messages",side_effect=error) as caller:
                    self.assertEqual(reference.main(args),2)
                    self.assertEqual(reference.main(args+["--resume"]),2)
                self.assertEqual(caller.call_count,1)
                self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["status"],"request_unknown")

    def test_definite_auth_failure_can_resume_own_history_without_gold(self):
        calls = []
        def call(_config, messages):
            calls.append(copy.deepcopy(messages))
            if len(calls) == 2:
                raise TeacherError("auth", http_status=401, category="auth")
            return {"text":"own answer", "usage":{}, "finish_reason":"end_turn", "reported_model":"actual"}
        with patch.object(build_demo_data,"ROOT",self.root),patch("teacher_api.load_config",return_value=self.config), \
             patch.object(reference,"_wallet",return_value=5),patch("teacher_api.call_teacher_messages",side_effect=call):
            self.assertEqual(reference.main(self.args()),2)
            self.assertEqual(reference.main(self.args("--resume")),0)
        self.assertEqual(len(calls),3)
        self.assertNotIn("PRIVATE GOLD",json.dumps(calls))
        self.assertNotIn("PRIVATE RUBRIC",json.dumps(calls))
        self.assertEqual(calls[2][-2:],[{"role":"assistant","content":"own answer"},{"role":"user","content":"revision"}])
        state=json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual((state["status"],state["completed_turns"],state["pending_turns"]),("complete",2,0))

    def test_output_saved_before_crash_is_reused_without_post(self):
        self.case["user_turns"] = []; self.write_cases([self.case])
        response={"text":"answer", "usage":{}, "finish_reason":"end_turn", "reported_model":"actual"}
        original=reference._atomic_text
        def interrupted(path, content):
            if path == self.state and json.loads(content).get("status") == "active" and json.loads(content).get("completed_turns") == 1:
                raise RuntimeError("simulated crash after reply saved")
            return original(path,content)
        with patch.object(build_demo_data,"ROOT",self.root),patch("teacher_api.load_config",return_value=self.config), \
             patch.object(reference,"_wallet",return_value=5),patch("teacher_api.call_teacher_messages",return_value=response) as caller:
            with patch.object(reference,"_atomic_text",side_effect=interrupted):
                with self.assertRaises(RuntimeError): reference.main(self.args())
            self.assertEqual(reference.main(self.args("--resume")),0)
        self.assertEqual(caller.call_count,1)

    def test_output_path_and_foreign_assistant_history_are_rejected(self):
        for kind in ("path", "history"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); private=root/"本地资料"; private.mkdir()
                case=copy.deepcopy(self.case)
                if kind=="history":
                    case["input_messages"]=[{"role":"user","content":"q"},{"role":"assistant","content":"foreign answer"},{"role":"user","content":"followup"}]
                source=private/"cases.jsonl";source.write_text(json.dumps(case)+"\n",encoding="utf-8")
                output=(root/"public.jsonl") if kind=="path" else private/"out.jsonl"
                args=["--cases",str(source),"--output",str(output),"--budget-state",str(private/"state.json"),"--use-account-balance"]
                with patch.object(build_demo_data,"ROOT",root),patch("teacher_api.load_config",return_value=self.config), \
                     patch.object(reference,"_wallet") as wallet,patch("teacher_api.call_teacher_messages") as caller:
                    self.assertEqual(reference.main(args),2)
                wallet.assert_not_called();caller.assert_not_called()

    def test_limit_resume_covers_full_case_set(self):
        first=copy.deepcopy(self.case);first['user_turns']=[]
        second=copy.deepcopy(first);second['case_id']='c2'
        self.write_cases([first,second])
        response={'text':'answer','usage':{},'finish_reason':'end_turn','reported_model':'actual'}
        with patch.object(build_demo_data,'ROOT',self.root),patch('teacher_api.load_config',return_value=self.config), \
             patch.object(reference,'_wallet',return_value=5),patch('teacher_api.call_teacher_messages',return_value=response) as caller:
            self.assertEqual(reference.main(self.args('--limit','1')),0)
            first_state=json.loads(self.state.read_text(encoding='utf-8'))
            self.assertEqual((first_state['total_cases'],first_state['completed_cases'],first_state['pending_turns']),(2,1,1))
            self.assertEqual(reference.main(self.args('--limit','1','--resume')),0)
        self.assertEqual(caller.call_count,2)
        self.assertEqual(json.loads(self.state.read_text(encoding='utf-8'))['status'],'complete')

    def test_insufficient_balance_is_recorded_without_guessing_from_status(self):
        error=TeacherError('safe',http_status=402,category='quota_billing',reason='insufficient_balance')
        with patch.object(build_demo_data,'ROOT',self.root),patch('teacher_api.load_config',return_value=self.config), \
             patch.object(reference,'_wallet',return_value=.1),patch('teacher_api.call_teacher_messages',side_effect=error):
            self.assertEqual(reference.main(self.args()),2)
        state=json.loads(self.state.read_text(encoding='utf-8'))
        self.assertEqual((state['status'],state['last_error_reason'],state['pending_turns']),
                         ('balance_insufficient','insufficient_balance',2))

    def test_global_lock_prevents_competing_reference_request(self):
        lock=self.private/'.formal_teacher_api.lock'
        lock.write_text('{"pid":1,"lock_id":"another-run"}',encoding='utf-8')
        with patch.object(build_demo_data,'ROOT',self.root),patch('teacher_api.load_config',return_value=self.config), \
             patch.object(reference,'_wallet') as wallet,patch('teacher_api.call_teacher_messages') as caller:
            self.assertEqual(reference.main(self.args()),2)
        wallet.assert_not_called();caller.assert_not_called()
        self.assertTrue(lock.exists())

    def test_reference_cannot_overwrite_cases_or_its_own_ledger(self):
        before=self.cases.read_bytes()
        for output, state in ((self.cases,self.state),(self.output,self.cases),(self.output,self.output)):
            args=['--cases',str(self.cases),'--output',str(output),'--budget-state',str(state),'--use-account-balance']
            with self.subTest(output=output,state=state),patch.object(build_demo_data,'ROOT',self.root), \
                 patch('teacher_api.load_config',return_value=self.config),patch.object(reference,'_wallet') as wallet:
                self.assertEqual(reference.main(args),2)
            wallet.assert_not_called()
            self.assertEqual(self.cases.read_bytes(),before)


if __name__ == "__main__":
    unittest.main()
