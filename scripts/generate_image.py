#!/usr/bin/env python3
"""通过可配置的 OpenAI Images 或 Gemini API 进行图片生成和编辑。"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import tempfile
import time
import uuid
import warnings
from contextlib import ExitStack
from datetime import datetime, timezone
from math import gcd, isfinite
from pathlib import Path
from typing import Any, Optional

# Keep the CLI importable through importlib without requiring installation as a package.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import requests
    from PIL import Image
    from filelock import FileLock, Timeout as LockTimeout
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Missing dependency: run 'python3 -m pip install -r requirements.txt'") from exc

from image_gen.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    MODEL_ID_RE,
    api_endpoint,
    api_headers,
    api_key_for,
    base_url,
    capability_profile,
    normalize_model,
    resolve_api_format,
    validate_api_url,
)
from image_gen.errors import ImageAPIError, error_payload
from image_gen.models import ProviderResponse
from image_gen.output_locks import acquire_output_locks, output_lock_paths
from image_gen.providers.gemini import call_gemini as provider_call_gemini
from image_gen.providers.openai import call_openai as provider_call_openai
from image_gen.responses import (
    MAX_REMOTE_IMAGE_BYTES,
    checked_json,
    download_image,
    extract_openai_image,
)
from image_gen.storage import pending_path, write_image_exclusive, write_json_atomic

DEFAULT_MODEL = DEFAULT_OPENAI_MODEL
GEMINI_MODEL = DEFAULT_GEMINI_MODEL
RATIO_RE = re.compile(r"^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$")
SIZE_RE = re.compile(r"^(\d+)[xX](\d+)$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SUPPORTED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MIME_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/webp": "webp",
}
MAX_IMAGE_PIXELS = 40_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env_file(path: Path) -> None:
    """加载常规 .env 文件，但不覆盖进程中已有的环境变量。"""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"无法读取环境配置文件 {path}：{exc}") from exc
    for line_number, original in enumerate(lines, 1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number} 不是有效的 .env 配置项")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not ENV_KEY_RE.fullmatch(key):
            raise ValueError(f"{path}:{line_number} 的环境变量名无效")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def load_environment(explicit_path: Optional[str]) -> list[str]:
    configured = explicit_path or os.getenv("IMAGE_API_ENV_FILE")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"环境配置文件不存在：{path}")
        load_env_file(path)
        return [str(path)]

    # 工作目录可能是不可信仓库，不能让它隐式改变凭据的接收端。
    candidates = [Path(__file__).resolve().parents[1] / ".env"]
    loaded: list[str] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file() and str(resolved) not in loaded:
            load_env_file(resolved)
            loaded.append(str(resolved))
    return loaded


def parse_ratio(value: str) -> tuple[float, float]:
    match = RATIO_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"宽高比 '{value}' 无效；应使用 W:H 格式，例如 16:9")
    width, height = (float(part) for part in match.groups())
    if not all(isfinite(v) and v > 0 for v in (width, height)) or max(width / height, height / width) > 3:
        raise ValueError("宽高比两边必须为正数，且比例不能超过 3:1")
    return width, height


def parse_exact_size(value: str) -> tuple[int, int]:
    match = SIZE_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"尺寸 '{value}' 无效；应使用 WIDTHxHEIGHT 格式")
    width, height = (int(part) for part in match.groups())
    if width < 256 or height < 256:
        raise ValueError("宽度和高度都不能小于 256 像素")
    if width % 16 or height % 16:
        raise ValueError("宽度和高度都必须是 16 的倍数")
    if max(width / height, height / width) > 3:
        raise ValueError("图片宽高比不能超过 3:1")
    return width, height


def ratio_for_size(width: int, height: int) -> str:
    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def size_for_ratio(aspect_ratio: str, image_size: str) -> str:
    ratio_width, ratio_height = parse_ratio(aspect_ratio)
    long_edge = {"1K": 1024, "2K": 2048, "4K": 4096}[image_size]
    if ratio_width >= ratio_height:
        width = long_edge
        height = round((long_edge * ratio_height / ratio_width) / 16) * 16
    else:
        height = long_edge
        width = round((long_edge * ratio_width / ratio_height) / 16) * 16
    return f"{max(width, 256)}x{max(height, 256)}"


def load_context(values: list[str], files: list[str]) -> list[str]:
    context = [value.strip() for value in values if value.strip()]
    for filename in files:
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"上下文文件不存在：{path}")
        text = path.read_text(encoding="utf-8").strip()
        if text:
            context.append(text)
    return context


def load_session(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取会话文件 {path}：{exc}") from exc
    if not isinstance(session, dict) or type(session.get("version")) is not int or session.get("version") != 1:
        raise ValueError(f"会话文件无效或版本不受支持：{path}")
    validate_session(session)
    return session


def validate_session(session: dict[str, Any]) -> None:
    """兼容缺省字段，但不把错误类型或无效枚举带入请求构造。"""
    def fail(field: str) -> None:
        raise ValueError(f"会话字段无效：{field}")

    if type(session.get("version")) is not int or session.get("version") != 1:
        fail("version")

    def parameters(value: Any, field: str) -> None:
        if not isinstance(value, dict):
            fail(field)
        for key, choices in {
            "quality": {"auto", "low", "medium", "high"},
            "output_format": {"png", "jpeg", "webp"},
            "image_size": {"1K", "2K", "4K"},
        }.items():
            if key in value and not (key == "image_size" and value[key] is None):
                if not isinstance(value[key], str) or value[key] not in choices:
                    fail(f"{field}.{key}")
        if "compression" in value and (
            type(value["compression"]) is not int or not 0 <= value["compression"] <= 100
        ):
            fail(f"{field}.compression")
        for key, parser in (("aspect_ratio", parse_ratio), ("size", parse_exact_size)):
            if key in value:
                if not isinstance(value[key], str):
                    fail(f"{field}.{key}")
                try:
                    parser(value[key])
                except (ValueError, OverflowError):
                    fail(f"{field}.{key}")

    if "model" in session:
        if not isinstance(session["model"], str) or not MODEL_ID_RE.fullmatch(session["model"]):
            fail("model")
    if "api_format" in session and session["api_format"] not in ("openai", "gemini"):
        fail("api_format")
    if "manifest" in session and (not isinstance(session["manifest"], str) or not session["manifest"] or "\x00" in session["manifest"]):
        fail("manifest")
    parameters(session.get("parameters", {}), "parameters")
    context = session.get("context", [])
    if not isinstance(context, list) or not all(isinstance(v, str) for v in context):
        fail("context")
    turns = session.get("turns", [])
    if not isinstance(turns, list):
        fail("turns")
    for index, turn in enumerate(turns):
        field = f"turns[{index}]"
        if not isinstance(turn, dict):
            fail(field)
        for key in ("prompt", "effective_prompt", "output_image", "input_image"):
            if key in turn and not (key == "input_image" and turn[key] is None):
                if not isinstance(turn[key], str) or "\x00" in turn[key]:
                    fail(f"{field}.{key}")
        if "input_images" in turn and (
            not isinstance(turn["input_images"], list)
            or not all(isinstance(v, str) and "\x00" not in v for v in turn["input_images"])
        ):
            fail(f"{field}.input_images")
        if "parameters" in turn:
            parameters(turn["parameters"], f"{field}.parameters")


def load_manifest(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        return {"version": 1, "created_at": utc_now(), "assets": {}}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 manifest {path}：{exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or not isinstance(manifest.get("assets"), dict)
    ):
        raise ValueError(f"manifest 无效或版本不受支持：{path}")
    return manifest


def manifest_file_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def manifest_record_path(manifest_path: Path, value: Path) -> str:
    """优先在 manifest 中保存可移植的相对路径。"""
    resolved = value.expanduser().resolve()
    try:
        return Path(os.path.relpath(resolved, manifest_path.parent)).as_posix()
    except ValueError:
        return str(resolved)


def validate_asset_id(value: str, label: str = "asset ID") -> str:
    if not ASSET_ID_RE.fullmatch(value):
        raise ValueError(
            f"{label} '{value}' 无效；仅允许字母、数字、点、下划线和连字符，最长 128 字符"
        )
    return value


def load_prompt(
    value: Optional[str], filename: Optional[str], manifest_path: Optional[Path]
) -> tuple[str, Optional[Path]]:
    if value and filename:
        raise ValueError("--prompt 与 --prompt-file 只能使用一个")
    if filename:
        path = (
            manifest_file_path(manifest_path, filename)
            if manifest_path
            else Path(filename).expanduser().resolve()
        )
        if not path.is_file():
            raise ValueError(f"提示词文件不存在：{path}")
        prompt = path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"提示词文件为空：{path}")
        return prompt, path
    if value and value.strip():
        return value.strip(), None
    raise ValueError("需要 --prompt 或 --prompt-file")


def last_output(session: dict[str, Any], session_path: Optional[Path] = None) -> Optional[Path]:
    for turn in reversed(session.get("turns", [])):
        output = turn.get("output_image")
        if output:
            return manifest_file_path(session_path, output) if session_path else Path(output).expanduser().resolve()
    return None


def compose_prompt(
    command: str,
    prompt: str,
    context: list[str],
    previous_turns: list[dict[str, Any]],
) -> str:
    labels = {
        "generate": "图片生成要求：",
        "reference": "结合视觉参考的新图片生成要求：",
        "edit": "图片编辑要求：",
    }
    sections = [labels[command], prompt.strip()]
    if context:
        sections.extend(["相关上下文：", "\n\n".join(context)])
    if command == "edit":
        if previous_turns:
            history = [turn.get("prompt", "").strip() for turn in previous_turns if turn.get("prompt")]
            if history:
                sections.extend(["此前的图片要求与编辑记录：", "\n".join(f"- {item}" for item in history)])
        sections.append("以提供的图片为编辑源图，并保持本轮未指定修改的元素不变。")
    elif command == "reference":
        sections.append(
            "提供的图片作为指定特征的视觉参考；按用户要求决定保留或改变构图、布局及其他特征。"
            "根据上述要求创作新图片。"
        )
    return "\n\n".join(sections)


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
    model: str = DEFAULT_MODEL,
) -> ProviderResponse:
    validate_api_url(url)
    return provider_call_openai(
        prompt,
        sources,
        size,
        quality,
        output_format,
        compression,
        key,
        url,
        timeout,
        model,
        endpoint_for=api_endpoint,
        headers_for=api_headers,
        source_mime=source_image_mime,
        checked_json=checked_json,
        extract_image=extract_openai_image,
    )


def call_gemini(
    prompt: str,
    sources: list[Path],
    aspect_ratio: str,
    image_size: str,
    key: str,
    url: str,
    timeout: int,
    model: str = GEMINI_MODEL,
) -> ProviderResponse:
    validate_api_url(url)
    return provider_call_gemini(
        prompt,
        sources,
        aspect_ratio,
        image_size,
        key,
        url,
        timeout,
        model,
        endpoint_for=api_endpoint,
        headers_for=api_headers,
        source_mime=source_image_mime,
        checked_json=checked_json,
    )


def canonical_extension(value: str) -> str:
    normalized = value.lower().lstrip(".")
    return "jpeg" if normalized == "jpg" else normalized


def detected_image_extension(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return None


def source_image_mime(path: Path) -> str:
    try:
        extension = validated_image_extension(path.read_bytes(), "application/octet-stream")
    except ImageAPIError as exc:
        raise ValueError(f"输入文件不是受支持的完整 PNG、JPEG 或 WebP 图片：{path}：{exc}") from exc
    return {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }[extension]


def validated_image_extension(data: bytes, mime: str) -> str:
    if not isinstance(data, bytes) or not isinstance(mime, str):
        raise ImageAPIError("API 图片字节或 MIME 类型无效", code="invalid_api_response", stage="response")
    if len(data) > MAX_REMOTE_IMAGE_BYTES:
        raise ImageAPIError("图片超过 32 MB 限制", code="invalid_api_response", stage="response")
    detected = detected_image_extension(data)
    if detected is None:
        raise ImageAPIError("API 返回的数据不是受支持的 PNG、JPEG 或 WebP 图片", code="invalid_api_response", stage="response")
    declared = MIME_EXTENSIONS.get(mime.split(";", 1)[0].lower())
    if declared and canonical_extension(declared) != canonical_extension(detected):
        raise ImageAPIError(f"API 返回的图片格式与 Content-Type 不一致：{mime} / {detected}", code="invalid_api_response", stage="response")
    image_dimensions(data)
    return detected


def image_dimensions(data: bytes) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ImageAPIError("图片像素数超过 4000 万限制", code="invalid_api_response", stage="response")
                dimensions = image.size
                image.verify()
            # verify 不会完整解码 JPEG 等格式，必须再次打开并 load。
            with Image.open(io.BytesIO(data)) as image:
                image.load()
        return dimensions
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageAPIError("图片数据损坏或无法完整解码", code="invalid_api_response", stage="response") from exc


def requested_output_path(output_dir: Path, requested: str, extension: str) -> Path:
    candidate = Path(requested).expanduser()
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    suffix = candidate.suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS or canonical_extension(suffix) != canonical_extension(extension):
        candidate = candidate.with_suffix(f".{extension}")
    return candidate.resolve()


def unique_output_path(output_dir: Path, requested: Optional[str], extension: str) -> Path:
    if requested:
        candidate = requested_output_path(output_dir, requested, extension)
        if candidate.exists():
            raise ValueError(f"拒绝覆盖已有输出文件：{candidate}")
        return candidate
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return (output_dir / f"image_{stamp}.{extension}").resolve()


def validate_requested_output(output_dir: Path, requested: Optional[str], extension: str) -> None:
    """在付费 API 调用前检查显式输出路径，避免调用完成后才发现冲突。"""
    if not requested:
        return
    candidate = requested_output_path(output_dir, requested, extension)
    if candidate.exists():
        raise ValueError(f"拒绝覆盖已有输出文件：{candidate}")


def command_input_paths(args: argparse.Namespace, manifest: Optional[Path]) -> list[Path]:
    """加锁前只解析路径，不读取内容，防止锁文件本身恰好是用户输入。"""
    image = getattr(args, "image", None)
    images = image if isinstance(image, list) else ([image] if image else [])
    inputs = [Path(value).expanduser().resolve() for value in images + args.context_file]
    if args.prompt_file:
        inputs.append(manifest_file_path(manifest, args.prompt_file) if manifest
                      else Path(args.prompt_file).expanduser().resolve())
    return inputs


def acquire_file_lock(
    stack: ExitStack, path: Path, dry_run: bool, inputs: Optional[list[Path]] = None
) -> None:
    protected = [path, pending_path(path), Path(str(path) + ".lock")]
    if any(p.resolve() in (inputs or []) for p in protected):
        raise ValueError("session/manifest 及其锁或恢复日志不能覆盖输入文件")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stack.enter_context(FileLock(str(path) + ".lock", timeout=30))
    except LockTimeout as exc:
        raise ValueError(f"文件正由另一个生成任务使用，请稍后重试：{path}") from exc


def reject_pending(path: Path) -> None:
    journal = pending_path(path)
    if journal.exists():
        raise ImageAPIError(
            f"存在待恢复的本地保存，禁止重复生图。请执行 recover --journal {journal}",
            code="storage_pending", stage="storage", recovery_journal=str(journal),
        )


def validate_paths(session: Path, manifest: Optional[Path], outputs: list[Path], inputs: list[Path]) -> None:
    metadata = [session] + ([manifest] if manifest else [])
    for path in metadata:
        if path.suffix.lower() != ".json" or path.name.endswith(".pending.json"):
            raise ValueError("session/manifest 必须使用非 .pending.json 的 JSON 路径")
    protected = metadata + [pending_path(p) for p in metadata] + [Path(str(p) + ".lock") for p in metadata]
    locks = output_lock_paths(outputs)
    if any(p in [v.resolve() for v in protected + inputs + outputs] for p in locks):
        raise ValueError("图片输出锁不能覆盖输入、输出、session 或 manifest")
    protected += locks
    resolved = [p.resolve() for p in protected]
    if len(set(resolved)) != len(resolved):
        raise ValueError("session、manifest 及恢复文件路径不能相同")
    for path in outputs:
        if path.resolve() in resolved or path.resolve() in inputs:
            raise ValueError("图片输出路径不能覆盖输入、session 或 manifest")
    if any(p in inputs for p in resolved):
        raise ValueError("session/manifest 不能覆盖输入文件")


def preflight_writable(paths: list[Path]) -> None:
    """在付费前试写目录；实际写入仍需处理权限变化和磁盘耗尽。"""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (not path.is_file() or not os.access(path, os.W_OK)):
            raise ValueError(f"目标文件不可写：{path}")
        with tempfile.TemporaryFile(dir=path.parent) as handle:
            handle.write(b"probe")
            handle.flush()


def preflight_image_directory(directory: Path) -> None:
    """提前确认目标文件系统支持排他发布所需的硬链接。"""
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as probe_directory:
        source = Path(probe_directory) / "source"
        source.write_bytes(b"probe")
        try:
            os.link(source, Path(probe_directory) / "link")
        except OSError as exc:
            raise ValueError("输出目录不支持硬链接排他发布，请使用本地 NTFS/ext4/APFS 等目录") from exc


def finish_transaction(transaction: dict[str, Any]) -> dict[str, Any]:
    result = transaction["result"]
    output = Path(result["image"])
    session = Path(result["session"])
    manifest = Path(result["manifest"]) if result["manifest"] else None
    data = base64.b64decode(transaction["image_base64"], validate=True)
    validated_image_extension(data, result["mime_type"])
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError(f"恢复目标已有不同内容，拒绝覆盖；请保留恢复日志：{output}")
    else:
        write_image_exclusive(output, data)
    write_json_atomic(session, transaction["session_payload"])
    if manifest:
        write_json_atomic(manifest, transaction["manifest_payload"])
    # 仅在所有数据落盘后解除阻塞；重复恢复不会追加第二个 turn。
    pending_path(session).unlink(missing_ok=True)
    if manifest:
        pending_path(manifest).unlink(missing_ok=True)
    return result


def save_transaction(result: dict[str, Any], session: dict[str, Any], manifest: Optional[dict[str, Any]], data: bytes) -> None:
    transaction = {
        "version": 1, "result": result, "session_payload": session,
        "manifest_payload": manifest, "image_base64": base64.b64encode(data).decode("ascii"),
    }
    session_path = Path(result["session"])
    manifest_path = Path(result["manifest"]) if result["manifest"] else None
    journals = ([pending_path(manifest_path)] if manifest_path else []) + [pending_path(session_path)]
    try:
        # 保存完整图片和待提交元数据后才改动成品；恢复过程无需重新调用 API。
        for journal in journals:
            write_json_atomic(journal, transaction)
        finish_transaction(transaction)
    except (OSError, ValueError) as exc:
        available = next((p for p in journals if p.exists()), None)
        if available:
            raise ImageAPIError(
                f"本地保存未完成；不要重新生成，请执行 recover --journal {available}：{exc}",
                code="storage_pending", stage="storage", provider=result.get("api_format"),
                http_status=result.get("http_status"), recovery_journal=str(available),
            ) from exc
        raise ImageAPIError(
            f"无法保存恢复日志，API 已返回但结果未持久化；不要自动重试付费请求：{exc}",
            code="storage_failed", stage="storage", provider=result.get("api_format"),
            http_status=result.get("http_status"),
        ) from exc


def recover_transaction(journal: Path, requested_output: Optional[str] = None) -> dict[str, Any]:
    def read_transaction(location: Path) -> dict[str, Any]:
        value = json.loads(location.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("恢复日志格式无效")
        result = value.get("result")
        if not isinstance(result, dict) or not all(isinstance(result.get(k), str) for k in ("image", "session", "mime_type")):
            raise ValueError("恢复日志 result 字段无效")
        if "manifest" not in result or (result["manifest"] is not None and not isinstance(result["manifest"], str)):
            raise ValueError("恢复日志 manifest 路径无效")
        for key in ("session", "image", "manifest"):
            if result[key] is not None and (not result[key] or not Path(result[key]).is_absolute() or "\x00" in result[key]):
                raise ValueError(f"恢复日志 {key} 必须是绝对路径")
        if not isinstance(value.get("session_payload"), dict) or not isinstance(value.get("image_base64"), str):
            raise ValueError("恢复日志 session/image 字段无效")
        validate_session(value["session_payload"])
        manifest = Path(result["manifest"]).resolve() if result.get("manifest") else None
        session = Path(result["session"]).resolve()
        validate_paths(session, manifest, [Path(result["image"]).resolve()], [])
        if location not in [pending_path(p) for p in (session, manifest) if p]:
            raise ValueError("恢复日志路径与记录不一致")
        if manifest and (not isinstance(value.get("manifest_payload"), dict)
                         or not isinstance(value["manifest_payload"].get("assets"), dict)):
            raise ValueError("恢复日志 manifest 内容无效")
        if last_output(value["session_payload"], session) != Path(result["image"]).resolve():
            raise ValueError("恢复日志 session 的成品路径不一致")
        if manifest:
            asset_id = result.get("asset_id")
            if not isinstance(asset_id, str):
                raise ValueError("恢复日志 asset_id 无效")
            record = value["manifest_payload"]["assets"].get(asset_id)
            if not isinstance(record, dict) or not all(isinstance(record.get(k), str) for k in ("image", "session")):
                raise ValueError("恢复日志资产记录无效")
            if manifest_file_path(manifest, record["session"]) != session or manifest_file_path(manifest, record["image"]) != Path(result["image"]).resolve():
                raise ValueError("恢复日志资产路径不一致")
        return value

    def authoritative_transaction() -> dict[str, Any]:
        value = read_transaction(journal)
        result = value["result"]
        # manifest 日志先提交；改存中断时，session 副本可能仍记录旧目标。
        if result.get("manifest"):
            primary = pending_path(Path(result["manifest"]))
            if primary != journal and primary.exists():
                latest = read_transaction(primary)
                if any(latest["result"].get(k) != result.get(k) for k in ("session", "manifest")):
                    raise ValueError("恢复日志副本的 session/manifest 不一致")
                return latest
        return value

    def transaction_inputs(value: dict[str, Any]) -> list[Path]:
        result = value["result"]
        session = Path(result["session"])
        manifest = Path(result["manifest"]) if result["manifest"] else None
        inputs = []
        for turn in value["session_payload"]["turns"]:
            inputs.extend(manifest_file_path(session, p) for p in turn.get("input_images", []))
            if turn.get("input_image"):
                inputs.append(manifest_file_path(session, turn["input_image"]))
        if manifest:
            for record in value["manifest_payload"]["assets"].values():
                if isinstance(record, dict) and record.get("prompt_file"):
                    prompt_file = record["prompt_file"]
                    if not isinstance(prompt_file, str) or "\x00" in prompt_file:
                        raise ValueError("恢复日志资产的 prompt_file 路径无效")
                    inputs.append(manifest_file_path(manifest, prompt_file))
        return inputs

    transaction = authoritative_transaction()
    with ExitStack() as stack:
        result = transaction["result"]
        inputs = transaction_inputs(transaction)
        validate_paths(Path(result["session"]), Path(result["manifest"]) if result["manifest"] else None,
                       [Path(result["image"])], inputs)
        if result.get("manifest"):
            acquire_file_lock(stack, Path(result["manifest"]), False, inputs)
        acquire_file_lock(stack, Path(result["session"]), False, inputs)
        current = authoritative_transaction()
        if current != transaction:
            raise ValueError("恢复日志已变化，请重新执行恢复")
        result = current["result"]
        session_path = Path(result["session"])
        manifest_path = Path(result["manifest"]) if result["manifest"] else None
        old_output = Path(result["image"])
        try:
            data = base64.b64decode(current["image_base64"], validate=True)
            extension = validated_image_extension(data, result["mime_type"])
        except (ValueError, ImageAPIError) as exc:
            raise ValueError(f"恢复日志中的图片数据无效：{exc}") from exc
        output = requested_output_path(journal.parent, requested_output, extension) if requested_output else old_output
        inputs = transaction_inputs(current)
        validate_paths(session_path, manifest_path, [old_output, output], inputs)
        acquire_output_locks(stack, [old_output, output], False)
        if output != old_output:
            if output.exists():
                raise ValueError(f"拒绝覆盖已有输出文件：{output}")
            preflight_writable([output])
            preflight_image_directory(output.parent)
            # 先更新日志再发布图片，任何一次中断都能按新目标继续恢复。
            result["image"] = str(output)
            result["requested_output"] = requested_output
            current["session_payload"]["turns"][-1]["output_image"] = manifest_record_path(session_path, output)
            if manifest_path:
                current["manifest_payload"]["assets"][result["asset_id"]]["image"] = manifest_record_path(manifest_path, output)
            save_transaction(result, current["session_payload"], current["manifest_payload"], data)
            return result
        try:
            return finish_transaction(current)
        except OSError as exc:
            available = pending_path(manifest_path or session_path)
            raise ImageAPIError(
                f"本地恢复未完成；不要重新生成，请执行 recover --journal {available}：{exc}",
                code="storage_pending", stage="storage", provider=result.get("api_format"),
                http_status=result.get("http_status"), recovery_journal=str(available),
            ) from exc


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", help="直接的生成或编辑指令")
    parser.add_argument("--prompt-file", help="包含完整生成或编辑指令的 UTF-8 文件")
    parser.add_argument("--context", action="append", default=[], help="相关上下文；可重复传入")
    parser.add_argument("--context-file", action="append", default=[], help="UTF-8 上下文文件；可重复传入")
    parser.add_argument("--model", help="覆盖所选 API 的模型 ID；默认读取 OPENAI_MODEL 或 GEMINI_MODEL")
    parser.add_argument("--api-format", choices=("openai", "gemini"), help="默认 openai；仅显式指定 gemini 时使用 Gemini 配置")
    parser.add_argument("--base-url", help="API 根地址或带 /v1、/v1beta 的地址；必须使用 HTTPS")
    parser.add_argument("--aspect-ratio", help="目标宽高比，例如 16:9（默认）")
    parser.add_argument("--image-size", choices=("1K", "2K", "4K"), help="图片尺寸档位（默认：2K）")
    parser.add_argument("--size", help="OpenAI Images 精确像素尺寸，例如 2048x1152")
    parser.add_argument("--quality", choices=("auto", "low", "medium", "high"), help="OpenAI Images 渲染质量（默认：auto）")
    parser.add_argument("--output-format", choices=("png", "jpeg", "webp"), help="OpenAI Images 输出格式")
    parser.add_argument("--compression", type=int, help="JPEG/WebP 压缩参数 0-100（默认：100）")
    parser.add_argument("--output-dir", help="输出图片和默认会话文件所在目录")
    parser.add_argument("--output", help="输出文件名或路径；不会覆盖已有文件")
    parser.add_argument("--session", help="要创建或继续使用的会话 JSON 路径")
    parser.add_argument("--manifest", help="generation-manifest.json 路径；需与 --asset-id 同时使用")
    parser.add_argument("--asset-id", help="manifest 中稳定且唯一的资产 ID")
    parser.add_argument("--asset-role", help="资产用途，例如 cover 或 content-card")
    parser.add_argument("--env-file", help="显式选择可信 .env 文件（默认只读取技能目录）")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP 超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="不访问 API，仅校验并输出请求计划")


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ImageAPIError(message, code="argument_error", stage="arguments")


def build_parser() -> argparse.ArgumentParser:
    parser = JSONArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="根据文本和上下文生成图片")
    add_common_options(generate)
    reference = subparsers.add_parser("reference", help="结合视觉参考生成一张新图片")
    add_common_options(reference)
    reference.add_argument("--image", action="append", default=[], help="参考图片；可重复传入")
    reference.add_argument("--reference-asset-id", help="从 manifest 读取作为视觉锚点的资产 ID")
    edit = subparsers.add_parser("edit", help="编辑图片并保留会话上下文")
    add_common_options(edit)
    edit.add_argument("--image", help="编辑源图；会话已有上次输出时可省略")
    edit.add_argument(
        "--allow-api-switch",
        action="store_true",
        help="显式允许使用与会话记录不同的 API 编辑图片",
    )
    doctor = subparsers.add_parser("doctor", help="检查所选 API 的本地配置，不联网")
    doctor.add_argument("--api-format", choices=("openai", "gemini"), help="默认检查 OpenAI 配置")
    doctor.add_argument("--env-file", help="显式选择可信 .env 文件（默认只读取技能目录）")
    doctor.add_argument("--base-url", help="覆盖所选 API 的地址；必须使用 HTTPS")
    doctor.add_argument("--model", help="覆盖所选 API 的模型 ID")
    recover = subparsers.add_parser("recover", help="恢复未完成的本地保存，不调用 API")
    recover.add_argument("--journal", required=True, help="错误消息中的 .pending.json 恢复日志")
    recover.add_argument("--output", help="将待恢复图片安全改存到新路径，并同步 session/manifest")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "recover":
        return recover_transaction(Path(args.journal).expanduser().resolve(), args.output)
    if args.command == "doctor":
        return run_doctor(args)
    with ExitStack() as stack:
        if args.manifest:
            manifest_path = Path(args.manifest).expanduser().resolve()
            acquire_file_lock(stack, manifest_path, args.dry_run, command_input_paths(args, manifest_path))
            reject_pending(manifest_path)
        return run_locked(args, stack)


def validate_capabilities(api_format: str, capabilities: Any) -> None:
    parse_ratio(capabilities.default_aspect_ratio)
    if capabilities.default_image_size not in {"1K", "2K", "4K"}:
        raise ValueError(
            f"{api_format.upper()}_DEFAULT_IMAGE_SIZE 必须是 1K、2K 或 4K"
        )
    if capabilities.default_exact_size:
        parse_exact_size(capabilities.default_exact_size)
    if api_format == "gemini" and "png" not in capabilities.supported_formats:
        raise ValueError("GEMINI_SUPPORTED_FORMATS 必须包含 png")
    prefix = api_format.upper()
    if capabilities.default_output_format not in capabilities.supported_formats:
        raise ValueError(f"{prefix}_DEFAULT_OUTPUT_FORMAT 不在 {prefix}_SUPPORTED_FORMATS 中")
    if capabilities.default_quality not in capabilities.supported_qualities:
        raise ValueError(f"{prefix}_DEFAULT_QUALITY 不在 {prefix}_SUPPORTED_QUALITIES 中")
    if api_format == "gemini" and (
        capabilities.default_output_format != "png" or capabilities.default_quality != "auto"
    ):
        raise ValueError("Gemini 默认输出格式与质量必须是 png 和 auto")


def run_doctor(args: argparse.Namespace) -> dict[str, Any]:
    loaded_env_files = load_environment(args.env_file)
    api_format = resolve_api_format(args.api_format)
    url = base_url(args.base_url, api_format)
    model = normalize_model(args.model, api_format)
    capabilities = capability_profile(api_format)
    validate_capabilities(api_format, capabilities)
    auth_headers = api_headers("configured", api_format)
    key, key_name = api_key_for(api_format)
    if not key:
        raise ValueError(f"缺少 API Key：请在环境变量中设置 {key_name}")
    return {
        "status": "ok",
        "api_format": api_format,
        "base_url": url,
        "model": model,
        "auth": "x-goog-api-key" if "x-goog-api-key" in auth_headers else "bearer",
        "key_configured": True,
        "env_files": loaded_env_files,
        "capabilities": capabilities.as_dict(),
    }


def normalize_provider_response(value: Any, provider: str) -> ProviderResponse:
    if isinstance(value, ProviderResponse):
        return value
    if isinstance(value, tuple) and len(value) == 3:
        image, mime_type, endpoint = value
        return ProviderResponse(image, mime_type, endpoint, 200)
    raise ImageAPIError(
        "API 适配器返回了无效结果",
        code="invalid_provider_response",
        provider=provider,
    )


def run_locked(args: argparse.Namespace, stack: ExitStack) -> dict[str, Any]:
    loaded_env_files = load_environment(args.env_file)

    if bool(args.manifest) != bool(args.asset_id):
        raise ValueError("--manifest 与 --asset-id 必须同时使用")
    if args.asset_role and not args.asset_id:
        raise ValueError("--asset-role 仅能与 --manifest 和 --asset-id 一起使用")

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    manifest = load_manifest(manifest_path)
    asset_id = validate_asset_id(args.asset_id) if args.asset_id else None
    reference_asset_id = getattr(args, "reference_asset_id", None)
    if reference_asset_id:
        validate_asset_id(reference_asset_id, "reference asset ID")
        if not manifest_path or manifest is None:
            raise ValueError("--reference-asset-id 需要同时使用 --manifest 和 --asset-id")
        if reference_asset_id == asset_id:
            raise ValueError("资产不能把自身作为 reference_asset_id")

    assets = manifest["assets"] if manifest is not None else {}
    existing_asset = assets.get(asset_id) if asset_id else None
    if existing_asset is not None and not isinstance(existing_asset, dict):
        raise ValueError(f"manifest 资产记录无效：{asset_id}")
    if args.command == "edit" and manifest is not None and existing_asset is None:
        raise ValueError(f"manifest 中不存在可编辑资产：{asset_id}")
    if args.command in {"generate", "reference"} and existing_asset is not None:
        raise ValueError(f"manifest 中的 asset_id 已存在：{asset_id}")
    if (
        existing_asset
        and args.asset_role
        and existing_asset.get("role")
        and existing_asset["role"] != args.asset_role
    ):
        raise ValueError(
            f"资产 {asset_id} 的 role 已是 {existing_asset['role']}，不能改为 {args.asset_role}"
        )

    explicit_session_path = Path(args.session).expanduser().resolve() if args.session else None
    if args.command == "edit" and existing_asset is not None:
        stored_session = existing_asset.get("session")
        if not isinstance(stored_session, str) or not stored_session:
            raise ValueError(f"manifest 资产 {asset_id} 没有可用的 session")
        assert manifest_path is not None
        manifest_session_path = manifest_file_path(manifest_path, stored_session)
        if explicit_session_path and explicit_session_path != manifest_session_path:
            raise ValueError("--session 与 manifest 中该资产的 session 不一致")
        session_path = manifest_session_path
    else:
        session_path = explicit_session_path

    if session_path is None:
        default_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
            manifest_path.parent if manifest_path else (Path.cwd() / "generated_images").resolve()
        )
        session_path = default_dir / f"session_{uuid.uuid4().hex}.json"
    validate_paths(session_path, manifest_path, [], [])
    acquire_file_lock(stack, session_path, args.dry_run, command_input_paths(args, manifest_path))
    reject_pending(session_path)
    session = load_session(session_path)
    if session and session.get("manifest"):
        owner = manifest_file_path(session_path, session["manifest"])
        if manifest_path != owner:
            raise ValueError(f"此 session 属于 manifest；请通过 --manifest {owner} --asset-id 编辑")
    if args.command == "edit" and existing_asset is not None and session is None:
        raise ValueError(f"manifest 资产 {asset_id} 的 session 文件不存在：{session_path}")

    prompt, prompt_file = load_prompt(args.prompt, args.prompt_file, manifest_path)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif session_path:
        output_dir = session_path.parent
    elif manifest_path:
        output_dir = manifest_path.parent
    else:
        output_dir = (Path.cwd() / "generated_images").resolve()
    if args.command in {"generate", "reference"} and session:
        raise ValueError(f"{args.command} 不能继续已有会话；请改用 edit")

    api_format = resolve_api_format(args.api_format)
    session_api_format = (session or {}).get("api_format", "openai")
    same_api = not session or session_api_format == api_format
    if (
        args.command == "edit"
        and session
        and not same_api
        and not getattr(args, "allow_api_switch", False)
    ):
        raise ValueError(
            f"会话使用 {session_api_format}，本次选择 {api_format}；"
            "如确认跨 API 编辑，请显式传入 --allow-api-switch"
        )
    previous_params = session.get("parameters", {}) if session and same_api else {}
    # 每次调用默认 OpenAI，不从模型名或历史会话推断 API。
    inherited_model = (session or {}).get("model") if same_api else None
    model = normalize_model(args.model or inherited_model, api_format)
    capabilities = capability_profile(api_format)
    validate_capabilities(api_format, capabilities)
    if args.size and (args.aspect_ratio or args.image_size):
        raise ValueError("--size 不能与 --aspect-ratio 或 --image-size 同时使用")
    if api_format == "gemini":
        if args.size:
            raise ValueError("--size 仅适用于 OpenAI Images；Gemini 请使用 --aspect-ratio 和 --image-size")
        if args.quality is not None:
            raise ValueError("--quality 仅适用于 OpenAI Images，Gemini 不支持")
        if args.compression is not None:
            raise ValueError("--compression 仅适用于 OpenAI Images，Gemini 不支持")
        if args.output_format and args.output_format != "png":
            raise ValueError("Gemini 适配器当前只输出 PNG，不支持其他 --output-format")

    previous_size = previous_params.get("size")
    if args.size:
        width, height = parse_exact_size(args.size)
        size = f"{width}x{height}"
        aspect_ratio = ratio_for_size(width, height)
        image_size: Optional[str] = None
    elif (
        api_format == "openai"
        and not args.aspect_ratio
        and not args.image_size
        and not previous_size
        and capabilities.default_exact_size
    ):
        width, height = parse_exact_size(capabilities.default_exact_size)
        size = f"{width}x{height}"
        aspect_ratio = ratio_for_size(width, height)
        image_size = None
    elif args.aspect_ratio or args.image_size or not previous_size:
        aspect_ratio = (
            args.aspect_ratio
            or previous_params.get("aspect_ratio")
            or capabilities.default_aspect_ratio
        )
        parse_ratio(aspect_ratio)
        image_size = (
            args.image_size
            or previous_params.get("image_size")
            or capabilities.default_image_size
        )
        size = size_for_ratio(aspect_ratio, image_size)
    else:
        size = str(previous_size)
        width, height = parse_exact_size(size)
        aspect_ratio = previous_params.get("aspect_ratio") or ratio_for_size(width, height)
        parse_ratio(aspect_ratio)
        image_size = previous_params.get("image_size")
    if api_format == "gemini" and image_size is None:
        image_size = args.image_size or capabilities.default_image_size

    quality = args.quality or previous_params.get("quality") or capabilities.default_quality
    output_format = (
        "png"
        if api_format == "gemini"
        else (args.output_format or previous_params.get("output_format") or capabilities.default_output_format)
    )
    if output_format not in capabilities.supported_formats:
        raise ValueError(
            f"{api_format} 不支持输出格式 {output_format}；"
            f"已配置：{', '.join(capabilities.supported_formats)}"
        )
    if quality not in capabilities.supported_qualities:
        raise ValueError(
            f"{api_format} 不支持质量 {quality}；"
            f"已配置：{', '.join(capabilities.supported_qualities)}"
        )
    compression = args.compression
    if compression is None:
        compression = previous_params.get("compression", 100)
    if not 0 <= compression <= 100:
        raise ValueError("压缩参数必须在 0 到 100 之间")
    if args.timeout <= 0:
        raise ValueError("超时时间必须是正数")

    new_context = load_context(args.context, args.context_file)
    recorded_context = list((session or {}).get("context", []))
    combined_context = recorded_context + [item for item in new_context if item not in recorded_context]
    previous_turns = list((session or {}).get("turns", []))
    effective_prompt = compose_prompt(args.command, prompt, combined_context, previous_turns)

    source_images: list[Path] = []
    direct_reference_images: list[Path] = []
    reference_asset_image: Optional[Path] = None
    if args.command == "edit":
        source = Path(args.image).expanduser().resolve() if args.image else (last_output(session, session_path) if session else None)
        if not source or not source.is_file():
            raise ValueError("edit 需要 --image，或需要一个包含已有输出图片的会话")
        source_images = [source]
    elif args.command == "reference":
        direct_reference_images = [Path(item).expanduser().resolve() for item in args.image]
        missing = [str(path) for path in direct_reference_images if not path.is_file()]
        if missing:
            raise ValueError(f"参考图片不存在：{missing[0]}")
        source_images = list(direct_reference_images)
        if reference_asset_id:
            reference_asset = assets.get(reference_asset_id)
            if not isinstance(reference_asset, dict):
                raise ValueError(f"manifest 中不存在参考资产：{reference_asset_id}")
            if reference_asset.get("status") not in {None, "generated", "complete"}:
                raise ValueError(f"参考资产尚未完成：{reference_asset_id}")
            stored_image = reference_asset.get("image")
            if not isinstance(stored_image, str) or not stored_image:
                raise ValueError(f"参考资产没有可用图片：{reference_asset_id}")
            assert manifest_path is not None
            reference_asset_image = manifest_file_path(manifest_path, stored_image)
            if not reference_asset_image.is_file():
                raise ValueError(f"参考资产图片不存在：{reference_asset_image}")
            if reference_asset_image not in source_images:
                source_images.append(reference_asset_image)
        if not source_images:
            raise ValueError("reference 需要 --image 或 --reference-asset-id")
    for source in source_images:
        if source.stat().st_size > capabilities.max_input_image_bytes:
            limit_mb = capabilities.max_input_image_bytes / (1024 * 1024)
            raise ValueError(f"输入图片超过 {api_format} 的 {limit_mb:g} MB 限制：{source}")
        source_image_mime(source)

    url = base_url(args.base_url, api_format)
    api_headers("", api_format)  # dry-run 也检查认证方式，但不读取或输出密钥。
    endpoint = api_endpoint(url, api_format, model, bool(source_images))
    if api_format == "gemini":
        parameters = {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "output_format": "png",
        }
    else:
        parameters = {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "compression": compression,
        }
    expected_extension = "png" if api_format == "gemini" else output_format
    inputs = source_images + ([prompt_file] if prompt_file else []) + [Path(p).expanduser().resolve() for p in args.context_file]
    candidates = [requested_output_path(output_dir, args.output, ext) for ext in ("png", "jpeg", "webp")] if args.output else []
    validate_paths(session_path, manifest_path, candidates, inputs)
    # 请求前预留所有可能的实际扩展名，独立会话也不能为同一输出重复付费。
    acquire_output_locks(stack, candidates, args.dry_run)
    for candidate in candidates:
        if candidate.exists():
            raise ValueError(f"拒绝覆盖已有输出文件：{candidate}")
    validate_requested_output(output_dir, args.output, expected_extension)
    if args.dry_run:
        return {
            "dry_run": True,
            "command": args.command,
            "model": model,
            "api_format": api_format,
            "endpoint": endpoint,
            "source_image": str(source_images[0]) if len(source_images) == 1 else None,
            "source_images": [str(source) for source in source_images],
            "session": str(session_path),
            "manifest": str(manifest_path) if manifest_path else None,
            "asset_id": asset_id,
            "asset_role": args.asset_role or (existing_asset or {}).get("role"),
            "reference_asset_id": reference_asset_id,
            "reference_asset_image": (
                str(reference_asset_image) if reference_asset_image else None
            ),
            "prompt_file": str(prompt_file) if prompt_file else None,
            "env_files": loaded_env_files,
            "capabilities": capabilities.as_dict(),
            "parameters": parameters,
            "planned_output": (
                str(requested_output_path(output_dir, args.output, expected_extension))
                if args.output
                else None
            ),
            "effective_prompt": effective_prompt,
        }

    key, key_name = api_key_for(api_format)
    if not key:
        raise ValueError(f"缺少 API Key：请在环境变量中设置 {key_name}")

    preflight_writable([session_path, pending_path(session_path)] + ([manifest_path, pending_path(manifest_path)] if manifest_path else []) + (candidates or [output_dir / "image.png"]))
    preflight_image_directory(candidates[0].parent if candidates else output_dir)

    started = time.monotonic()
    if api_format == "gemini":
        provider_result = normalize_provider_response(
            call_gemini(
                effective_prompt,
                source_images,
                aspect_ratio,
                image_size,
                key,
                url,
                args.timeout,
                model=model,
            ),
            api_format,
        )
    else:
        provider_result = normalize_provider_response(
            call_openai(
                effective_prompt,
                source_images,
                size,
                quality,
                output_format,
                compression,
                key,
                url,
                args.timeout,
                model=model,
            ),
            api_format,
        )
    elapsed = round(time.monotonic() - started, 3)
    image_data = provider_result.image
    mime = provider_result.mime_type
    endpoint = provider_result.endpoint
    try:
        extension = validated_image_extension(image_data, mime)
        width, height = image_dimensions(image_data)
    except ImageAPIError as exc:
        exc.provider = exc.provider or api_format
        exc.http_status = provider_result.http_status if exc.http_status is None else exc.http_status
        raise
    output_path = unique_output_path(output_dir, args.output, extension)
    validate_paths(session_path, manifest_path, [output_path], inputs)

    if session is None:
        session = {
            "version": 1,
            "id": str(uuid.uuid4()),
            "created_at": utc_now(),
            "context": combined_context,
            "turns": [],
        }
    session.update({
        "updated_at": utc_now(),
        "model": model,
        "api_format": api_format,
        "base_url": url,
        "context": combined_context,
        "parameters": parameters,
    })
    if manifest_path:
        session["manifest"] = manifest_record_path(session_path, manifest_path)
    # 旧 session 的绝对路径仍可读取；新记录以 session 为基准保存相对路径。
    session.setdefault("turns", []).append({
        "type": args.command,
        "created_at": utc_now(),
        "prompt": prompt,
        "effective_prompt": effective_prompt,
        "input_image": manifest_record_path(session_path, source_images[0]) if args.command == "edit" else None,
        "input_images": [manifest_record_path(session_path, source) for source in source_images],
        "output_image": manifest_record_path(session_path, output_path),
        "actual_dimensions": {"width": width, "height": height},
        "mime_type": mime,
        "parameters": parameters,
        "http_status": provider_result.http_status,
        "request_id": provider_result.request_id,
        "usage": provider_result.usage,
        "elapsed_seconds": elapsed,
    })

    if manifest is not None:
        assert manifest_path is not None and asset_id is not None
        record = dict(existing_asset or {})
        record.setdefault("created_at", utc_now())
        role = args.asset_role or record.get("role")
        if role:
            record["role"] = role
        if prompt_file:
            record["prompt_file"] = manifest_record_path(manifest_path, prompt_file)
            record.pop("prompt", None)
        else:
            record["prompt"] = prompt
            record.pop("prompt_file", None)
        record.update({
            "action": args.command,
            "image": manifest_record_path(manifest_path, output_path),
            "session": manifest_record_path(manifest_path, session_path),
            "model": model,
            "api_format": api_format,
            "parameters": parameters,
            "status": "generated",
            "actual_dimensions": {"width": width, "height": height},
            "http_status": provider_result.http_status,
            "request_id": provider_result.request_id,
            "usage": provider_result.usage,
            "elapsed_seconds": elapsed,
            "updated_at": utc_now(),
        })
        if args.command == "reference":
            record["reference_images"] = [
                manifest_record_path(manifest_path, path) for path in direct_reference_images
            ]
            if reference_asset_id:
                record["reference_asset_id"] = reference_asset_id
            else:
                record.pop("reference_asset_id", None)
        assets[asset_id] = record
        manifest["updated_at"] = utc_now()

    result = {
        "image": str(output_path),
        "requested_output": args.output,
        "session": str(session_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "asset_id": asset_id,
        "model": model,
        "api_format": api_format,
        "endpoint": endpoint,
        "env_files": loaded_env_files,
        "parameters": parameters,
        "mime_type": mime,
        "bytes": len(image_data),
        "elapsed_seconds": elapsed,
        "http_status": provider_result.http_status,
        "request_id": provider_result.request_id,
        "usage": provider_result.usage,
        "actual_dimensions": {"width": width, "height": height},
        "status": "generated",
    }
    save_transaction(result, session, manifest, image_data)
    return result


def main() -> int:
    parser = build_parser()
    args = None
    try:
        args = parser.parse_args()
        result = run(args)
    except (ValueError, OSError, requests.RequestException, ImageAPIError) as exc:
        provider = (
            None
            if args is None or args.command == "recover"
            else resolve_api_format(getattr(args, "api_format", None))
        )
        print(json.dumps(error_payload(exc, provider), ensure_ascii=False), file=sys.stderr)
        return 2 if isinstance(exc, ImageAPIError) and exc.code == "argument_error" else 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
