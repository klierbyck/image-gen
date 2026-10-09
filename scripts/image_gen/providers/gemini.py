from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable

import requests

from ..models import ProviderResponse
from ..responses import decode_image_base64, invalid_response, validate_image_mime


def call_gemini(
    prompt: str,
    sources: list[Path],
    aspect_ratio: str,
    image_size: str,
    key: str,
    url: str,
    timeout: int,
    model: str,
    *,
    endpoint_for: Callable[[str, str, str, bool], str],
    headers_for: Callable[[str, str], dict[str, str]],
    source_mime: Callable[[Path], str],
    checked_json: Callable[[requests.Response, str], dict[str, Any]],
) -> ProviderResponse:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for source in sources:
        parts.append({
            "inlineData": {
                "mimeType": source_mime(source),
                "data": base64.b64encode(source.read_bytes()).decode("ascii"),
            }
        })
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": aspect_ratio, "imageSize": image_size},
        },
    }
    endpoint = endpoint_for(url, "gemini", model, bool(sources))
    response = requests.post(
        endpoint,
        headers={**headers_for(key, "gemini"), "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
        allow_redirects=False,
    )
    result = checked_json(response, "gemini")
    if not isinstance(result, dict):
        raise invalid_response("API 响应的根节点不是对象", "gemini")
    candidates = result.get("candidates", [])
    if not isinstance(candidates, list):
        raise invalid_response("API 响应的 candidates 不是列表", "gemini")
    # HTTP 200 和合法 JSON 并不保证嵌套结构正确；逐层验证，不捕获编程错误。
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise invalid_response("API 响应的 candidate 不是对象", "gemini")
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            raise invalid_response("API 响应的 content 不是对象", "gemini")
        response_parts = content.get("parts", [])
        if not isinstance(response_parts, list):
            raise invalid_response("API 响应的 parts 不是列表", "gemini")
        for part in response_parts:
            if not isinstance(part, dict):
                raise invalid_response("API 响应的 part 不是对象", "gemini")
            inline_field = "inlineData" if "inlineData" in part else "inline_data"
            if inline_field in part:
                inline = part[inline_field]
                if not isinstance(inline, dict):
                    raise invalid_response("API 响应的 inlineData 不是对象", "gemini")
                mime = validate_image_mime(
                    inline.get("mimeType", inline.get("mime_type", "image/png")), "gemini"
                )
                image = decode_image_base64(inline.get("data"), "gemini")
                usage = result.get("usageMetadata") or result.get("usage")
                return ProviderResponse(
                    image=image,
                    mime_type=mime,
                    endpoint=endpoint,
                    http_status=response.status_code,
                    request_id=response.headers.get("x-request-id") or response.headers.get("request-id"),
                    usage=usage if isinstance(usage, dict) else None,
                )
    raise invalid_response("API 响应中没有图片数据", "gemini")
