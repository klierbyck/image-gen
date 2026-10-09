from __future__ import annotations

import base64
from typing import Any, Optional
from urllib.parse import urlsplit

import requests

from .config import api_headers
from .errors import ImageAPIError


MAX_REMOTE_IMAGE_BYTES = 32 * 1024 * 1024
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}


def invalid_response(message: str, provider: Optional[str] = None) -> ImageAPIError:
    return ImageAPIError(
        message, code="invalid_api_response", stage="response", provider=provider
    )


def validate_image_mime(value: Any, provider: Optional[str] = None) -> str:
    if not isinstance(value, str):
        raise invalid_response("API 返回的图片 MIME 类型不是字符串", provider)
    mime = value.strip().lower()
    if mime not in IMAGE_MIME_TYPES:
        raise invalid_response("API 返回的图片 MIME 类型不受支持", provider)
    return mime


def decode_image_base64(value: Any, provider: Optional[str] = None) -> bytes:
    if not isinstance(value, str) or not value:
        raise invalid_response("API 返回的 base64 图片数据不是非空字符串", provider)
    # 先检查编码长度，避免在完整解码之前分配超出图片上限的内存。
    if len(value) > 4 * ((MAX_REMOTE_IMAGE_BYTES + 2) // 3):
        raise invalid_response("API 返回的图片超过 32 MB 限制", provider)
    try:
        image = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise invalid_response("API 返回的 base64 图片数据无效", provider) from exc
    if not image or len(image) > MAX_REMOTE_IMAGE_BYTES:
        raise invalid_response("API 返回的图片为空或超过 32 MB 限制", provider)
    return image


def error_text(response: requests.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    except (ValueError, TypeError):
        pass
    return response.text[:1000] or str(response.status_code)


def checked_json(
    response: requests.Response, provider: Optional[str] = None
) -> dict[str, Any]:
    if response.status_code < 200 or response.status_code >= 300:
        raise ImageAPIError(
            f"HTTP {response.status_code}：{error_text(response)}",
            code="api_http_error",
            provider=provider,
            retryable=response.status_code == 429 or response.status_code >= 500,
            http_status=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ImageAPIError(
            f"API 响应不是 JSON：{response.text[:500]}",
            code="invalid_api_response",
            stage="response",
            provider=provider,
        ) from exc
    if not isinstance(payload, dict):
        raise ImageAPIError(
            "API 响应的根节点不是对象",
            code="invalid_api_response",
            stage="response",
            provider=provider,
        )
    return payload


def decode_data_url(
    value: Any, provider: Optional[str] = None
) -> Optional[tuple[bytes, str]]:
    if not isinstance(value, str):
        raise invalid_response("API 返回的图片数据不是字符串", provider)
    if value[:5].lower() != "data:":
        return None
    header, separator, encoded = value.partition(",")
    metadata = header[5:].split(";")
    # 保留合法 MIME 参数，例如 charset；base64 必须是最后的编码标记。
    if (
        not separator or len(metadata) < 2 or metadata[-1].lower() != "base64"
        or any("=" not in parameter for parameter in metadata[1:-1])
    ):
        raise invalid_response("API 返回的图片 data URL 必须使用 base64 编码", provider)
    mime = validate_image_mime(metadata[0], provider)
    return decode_image_base64(encoded, provider), mime


def origin(value: str) -> tuple[str, str, Optional[int]]:
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    port = parsed.port or ({"http": 80, "https": 443}.get(scheme))
    return scheme, (parsed.hostname or "").lower(), port


def download_image(
    image_url: str,
    api_base_url: str,
    key: str,
    timeout: int,
) -> tuple[bytes, str]:
    # 返回的 URL 属于不可信响应；URL 语法问题应报告为响应错误。
    if not isinstance(image_url, str) or not image_url:
        raise invalid_response("API 返回的图片 URL 不是非空字符串", "openai")
    try:
        parsed = urlsplit(image_url)
        same_origin = origin(image_url) == origin(api_base_url)
    except ValueError as exc:
        raise invalid_response("API 返回的图片 URL 无效", "openai") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise invalid_response("图片下载地址不是有效的 HTTP(S) URL", "openai")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise invalid_response("图片下载地址不能包含用户名、密码或片段", "openai")
    if parsed.scheme != "https" and not (
        same_origin and parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ImageAPIError(
            "拒绝通过非 HTTPS 地址下载远程图片",
            code="image_download_url_error", stage="download", provider="openai",
        )
    headers = api_headers(key, "openai") if same_origin else {}
    response = None
    try:
        response = requests.get(
            image_url,
            headers=headers,
            timeout=timeout,
            stream=True,
            allow_redirects=False,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise ImageAPIError(
                f"图片下载失败，HTTP {response.status_code}：{error_text(response)}。"
                "仅重试图片下载，不要重新调用生图 API",
                code="image_download_http_error", stage="download", provider="openai",
                retryable=response.status_code == 429 or response.status_code >= 500,
                http_status=response.status_code,
            )
        final_url = getattr(response, "url", image_url)
        final_parsed = urlsplit(final_url)
        if final_parsed.scheme != "https" and not (
            origin(final_url) == origin(api_base_url)
            and final_parsed.hostname
            and final_parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ImageAPIError(
                "拒绝通过非 HTTPS 重定向下载远程图片",
                code="image_download_url_error", stage="download", provider="openai",
                http_status=response.status_code,
            )
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > MAX_REMOTE_IMAGE_BYTES:
                    raise ImageAPIError(
                        "远程图片超过 32 MB 限制", code="image_download_size_error",
                        stage="download", provider="openai", http_status=response.status_code,
                    )
            except ValueError:
                pass
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_REMOTE_IMAGE_BYTES:
                raise ImageAPIError(
                    "远程图片超过 32 MB 限制", code="image_download_size_error",
                    stage="download", provider="openai", http_status=response.status_code,
                )
            chunks.append(chunk)
        content_type = response.headers.get("Content-Type", "application/octet-stream")
        if not isinstance(content_type, str):
            raise invalid_response("图片下载响应的 MIME 类型不是字符串", "openai")
        mime = content_type.split(";", 1)[0]
        # 缺省或二进制类型仍由后续图片解码判断；显式声明的图片类型必须合法。
        if mime.strip().lower() != "application/octet-stream":
            mime = validate_image_mime(mime, "openai")
        return b"".join(chunks), mime
    except requests.RequestException as exc:
        # 此阶段 API 已生成图片，重试范围仅为下载，不能重复整个付费请求。
        raise ImageAPIError(
            f"图片下载网络失败：{exc}。仅重试图片下载，不要重新调用生图 API",
            code="image_download_timeout" if isinstance(exc, requests.Timeout) else "image_download_network_error",
            stage="download", provider="openai", retryable=True,
            http_status=response.status_code if response is not None else None,
        ) from exc
    finally:
        if response is not None:
            response.close()


def extract_openai_image(
    payload: dict[str, Any],
    key: str,
    timeout: int,
    expected_mime: str,
    api_base_url: str,
) -> tuple[bytes, str]:
    if not isinstance(payload, dict):
        raise invalid_response("API 响应的根节点不是对象", "openai")
    data = payload.get("data")
    if not isinstance(data, list):
        raise invalid_response("图片响应中没有 data 列表", "openai")
    for item in data:
        if not isinstance(item, dict):
            raise invalid_response("图片响应的 data 元素不是对象", "openai")
        for field in ("b64_json", "base64", "image_b64"):
            encoded = item.get(field)
            # 兼容只返回 URL 的网关：可选的 base64 字段允许为 null。
            if encoded is not None:
                decoded = decode_data_url(encoded, "openai")
                if decoded:
                    return decoded
                return (
                    decode_image_base64(encoded, "openai"),
                    validate_image_mime(expected_mime, "openai"),
                )
        url = item.get("url")
        if url is not None:
            return download_image(url, api_base_url, key, timeout)
    raise invalid_response("API 响应中没有图片数据", "openai")
