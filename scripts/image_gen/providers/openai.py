from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import requests

from ..models import ProviderResponse


def call_openai(
    prompt: str,
    sources: list[Path],
    size: str,
    quality: str,
    output_format: str,
    compression: int,
    key: str,
    url: str,
    timeout: int,
    model: str,
    *,
    endpoint_for: Callable[[str, str, str, bool], str],
    headers_for: Callable[[str, str], dict[str, str]],
    source_mime: Callable[[Path], str],
    checked_json: Callable[[requests.Response, str], dict[str, Any]],
    extract_image: Callable[[dict[str, Any], str, int, str, str], tuple[bytes, str]],
) -> ProviderResponse:
    fields: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "output_format": output_format,
    }
    if output_format in {"jpeg", "webp"}:
        fields["output_compression"] = compression
    headers = headers_for(key, "openai")
    endpoint = endpoint_for(url, "openai", model, bool(sources))
    if sources:
        with ExitStack() as stack:
            files = []
            for source in sources:
                image_file = stack.enter_context(source.open("rb"))
                files.append(("image", (source.name, image_file, source_mime(source))))
            response = requests.post(
                endpoint,
                headers=headers,
                data={name: str(value) for name, value in fields.items()},
                files=files,
                timeout=timeout,
                allow_redirects=False,
            )
    else:
        response = requests.post(
            endpoint,
            headers={**headers, "Content-Type": "application/json"},
            json=fields,
            timeout=timeout,
            allow_redirects=False,
        )
    payload = checked_json(response, "openai")
    image, mime = extract_image(payload, key, timeout, f"image/{output_format}", url)
    return ProviderResponse(
        image=image,
        mime_type=mime,
        endpoint=endpoint,
        http_status=response.status_code,
        request_id=response.headers.get("x-request-id") or response.headers.get("request-id"),
        usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
    )

