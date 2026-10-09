from __future__ import annotations

import os
import re
from typing import Any, Optional
from urllib.parse import quote, quote_plus

import requests


class ImageAPIError(RuntimeError):
    """可由调用方处理的 API 或响应错误。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "api_error",
        stage: str = "api",
        provider: Optional[str] = None,
        retryable: bool = False,
        http_status: Optional[int] = None,
        recovery_journal: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.provider = provider
        self.retryable = retryable
        self.http_status = http_status
        self.recovery_journal = recovery_journal


def redact_error(message: str) -> str:
    """在输出边界脱敏，避免网关回显的凭据进入日志或聊天。"""
    secrets = {os.getenv(name) for name in ("OPENAI_API_KEY", "GEMINI_API_KEY")}
    variants = {variant for secret in secrets if secret
                for variant in (secret, quote(secret, safe=""), quote_plus(secret))}
    for secret in sorted(variants, key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    # 同时处理未知凭据的 header、Bearer 及 URL query，保留非敏感诊断内容。
    message = re.sub(r"(?i)\bBearer\s+[^\s\"'<>;,}]+", "Bearer [REDACTED]", message)
    # Authorization 可以包含 Basic/Digest 等多段值，整行掩码以免残留认证内容。
    message = re.sub(
        r"(?im)(\bauthorization\b[\"']?\s*[:=]\s*)[^\r\n]+",
        r"\1[REDACTED]", message,
    )
    message = re.sub(
        r"(?i)(\b(?:x-goog-api-key|api[-_]?key|access[-_]?token|password)\b[\"']?\s*[:=]\s*[\"']?)([^\s\"'<>;,}&]+)",
        r"\1[REDACTED]", message,
    )
    message = re.sub(
        r"(?i)([?&](?:key|api[-_]?key|token|access[-_]?token|secret|password|signature|sig)=)[^\s&#\"'<>]+",
        r"\1[REDACTED]", message,
    )
    message = re.sub(r"(?i)(https?://)[^/\s@]+@", r"\1[REDACTED]@", message)
    return message


def error_payload(exc: Exception, provider: Optional[str] = None) -> dict[str, Any]:
    if isinstance(exc, ImageAPIError):
        code = exc.code
        stage = exc.stage
        retryable = exc.retryable
        http_status = exc.http_status
        provider = exc.provider or provider
    elif isinstance(exc, requests.Timeout):
        code, stage, retryable, http_status = "network_timeout", "network", True, None
    elif isinstance(exc, requests.RequestException):
        code, stage, retryable, http_status = "network_error", "network", True, None
    elif isinstance(exc, OSError):
        code, stage, retryable, http_status = "filesystem_error", "storage", False, None
    else:
        code, stage, retryable, http_status = "validation_error", "validation", False, None
    payload = {
        "error": redact_error(str(exc)),
        "error_code": code,
        "stage": stage,
        "provider": provider,
        "retryable": retryable,
        "http_status": http_status,
    }
    if isinstance(exc, ImageAPIError) and exc.recovery_journal:
        payload["recovery_journal"] = exc.recovery_journal
    return payload
