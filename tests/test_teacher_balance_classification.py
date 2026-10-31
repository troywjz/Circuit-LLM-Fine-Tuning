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


def raw_classify(status: int, body: bytes):
    error = HTTPError("https://placeholder.invalid/v1", status, "private reason", {},
                      io.BytesIO(body))
    return _classify_http_error(error)


class EdgeGatewayErrorTests(unittest.TestCase):
    """Cloudflare 边缘错误页代表请求没进入模型，与网关回传的 5xx 必须分开。"""

    def test_cloudflare_json_error_page_is_treated_as_unreachable(self):
        # 中转站源站不可达时，Cloudflare 会回这样一页 JSON 说明，正文里没有模型错误信息。
        body = (b'{"type":"https://developers.cloudflare.com/support/troubleshooting/'
                b'http-status-codes/cloudflare-5xx-errors/error-502/","title":'
                b'"Error 502: Bad gateway","status":502}')
        error = raw_classify(502, body)
        self.assertEqual(error.reason, "upstream_unreachable")

    def test_cloudflare_plain_text_error_page_is_treated_as_unreachable(self):
        error = raw_classify(502, b"error code: 502\n")
        self.assertEqual(error.reason, "upstream_unreachable")

    def test_gateway_json_502_stays_unclassified(self):
        # 网关自己回传的 JSON 502 可能已在上游处理过，不能标成"没送达"而重发。
        error = classify(502, "server_error")
        self.assertIsNone(error.reason)


if __name__ == "__main__":
    unittest.main()
