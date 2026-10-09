from __future__ import annotations

import os
import re
from typing import Optional
from urllib.parse import quote, urlsplit

from .models import CapabilityProfile


DEFAULT_OPENAI_MODEL = "gpt-image-2"
DEFAULT_GEMINI_MODEL = "nana-banana-2"
DEFAULT_MODELS = {"openai": DEFAULT_OPENAI_MODEL, "gemini": DEFAULT_GEMINI_MODEL}
MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")


def resolve_api_format(value: Optional[str]) -> str:
    configured = value or "openai"
    if configured not in {"openai", "gemini"}:
        raise ValueError("--api-format 必须是 openai 或 gemini")
    return configured


def normalize_model(value: Optional[str], api_format: str = "openai") -> str:
    raw = (value or os.getenv(f"{api_format.upper()}_MODEL") or DEFAULT_MODELS[api_format]).strip()
    if not MODEL_ID_RE.fullmatch(raw):
        raise ValueError("模型 ID 无效；请填写 API 提供的模型标识")
    return raw


def api_endpoint(url: str, api_format: str, model: str, has_images: bool) -> str:
    version = "v1beta" if api_format == "gemini" else "v1"
    root = url.rstrip("/")
    versions = ("/v1beta", "/v1") if api_format == "gemini" else ("/v1",)
    if not root.endswith(versions):
        root = f"{root}/{version}"
    if api_format == "gemini":
        return f"{root}/models/{quote(model, safe='')}:generateContent"
    return f"{root}/images/{'edits' if has_images else 'generations'}"


def api_headers(key: str, api_format: str) -> dict[str, str]:
    auth_name = f"{api_format.upper()}_API_AUTH"
    auth = os.getenv(auth_name) or "bearer"
    if auth == "bearer":
        return {"Authorization": f"Bearer {key}"}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    raise ValueError(f"{auth_name} 必须是 bearer 或 x-goog-api-key")


def api_key_for(api_format: str) -> tuple[Optional[str], str]:
    key_name = f"{api_format.upper()}_API_KEY"
    return os.getenv(key_name), key_name


def validate_api_url(value: str) -> None:
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("API 地址端口无效") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and port == 0)
    ):
        raise ValueError("API 地址必须是无用户名、密码、查询参数和片段的 HTTPS URL")


def base_url(explicit: Optional[str] = None, api_format: str = "openai") -> str:
    url_name = f"{api_format.upper()}_API_BASE_URL"
    value = (explicit or os.getenv(url_name) or "").strip().rstrip("/")
    if not value:
        raise ValueError(f"缺少 API 地址：请设置 {url_name} 或 --base-url")
    validate_api_url(value)
    return value


def _csv(name: str, default: tuple[str, ...], allowed: set[str]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = tuple(dict.fromkeys(item.strip().lower() for item in raw.split(",") if item.strip()))
    if not values or any(item not in allowed for item in values):
        raise ValueError(f"{name} 包含不支持的值；可选值：{', '.join(sorted(allowed))}")
    return values


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是正整数") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return value


def capability_profile(api_format: str) -> CapabilityProfile:
    prefix = api_format.upper()
    formats = _csv(
        f"{prefix}_SUPPORTED_FORMATS",
        ("png",) if api_format == "gemini" else ("png", "jpeg", "webp"),
        {"png", "jpeg", "webp"},
    )
    qualities = _csv(
        f"{prefix}_SUPPORTED_QUALITIES",
        ("auto",) if api_format == "gemini" else ("auto", "low", "medium", "high"),
        {"auto", "low", "medium", "high"},
    )
    exact = (os.getenv(f"{prefix}_DEFAULT_SIZE") or "").strip() or None
    return CapabilityProfile(
        default_aspect_ratio=(os.getenv(f"{prefix}_DEFAULT_ASPECT_RATIO") or "16:9").strip(),
        default_image_size=(os.getenv(f"{prefix}_DEFAULT_IMAGE_SIZE") or "2K").strip().upper(),
        default_exact_size=exact,
        supported_formats=formats,
        supported_qualities=qualities,
        max_input_image_bytes=_positive_int(f"{prefix}_MAX_INPUT_IMAGE_MB", 4) * 1024 * 1024,
        default_output_format=(os.getenv(f"{prefix}_DEFAULT_OUTPUT_FORMAT") or "png").strip().lower(),
        default_quality=(os.getenv(f"{prefix}_DEFAULT_QUALITY") or "auto").strip().lower(),
    )

