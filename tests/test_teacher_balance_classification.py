"""教师接口余额、限流与认证错误的离线分类测试。"""
from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from teacher_api import _classify_http_error


def classify(status: int, code: str):
    # 服务原文包含额外信息，程序只返回安全类别，不回显原文。
    body = json.dumps({"error": {"code": code, "message": "private diagnostic"}}).encode("utf-8")
    error = HTTPError("https://placeholder.invalid/v1", status, "private reason", {}, io.BytesIO(body))
    return _classify_http_error(error)


class TeacherBalanceClassificationTests(unittest.TestCase):
    def test_explicit_balance_error_is_distinguished(self):
        error = classify(429, "insufficient_balance")
        self.assertEqual(error.category, "quota_billing")
        self.assertEqual(error.reason, "insufficient_balance")
        self.assertNotIn("private diagnostic", str(error))

    def test_rate_limit_and_auth_do_not_mean_empty_balance(self):
        rate = classify(429, "rate_limit_exceeded")
        auth = classify(401, "invalid_api_key")
        self.assertEqual((rate.category, rate.reason), ("rate_limit", None))
        self.assertEqual((auth.category, auth.reason), ("auth", None))

    def test_opaque_payment_error_does_not_claim_exact_balance(self):
        error = classify(402, "unknown")
        self.assertEqual(error.category, "quota_billing")
        self.assertIsNone(error.reason)


if __name__ == "__main__":
    unittest.main()
