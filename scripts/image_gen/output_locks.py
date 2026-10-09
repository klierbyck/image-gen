"""请求前锁定显式输出，避免独立会话并发为同一路径重复付费。"""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock, Timeout as LockTimeout


OUTPUT_LOCK_SUFFIX = ".image-gen-output.lock"


def output_lock_paths(paths: Iterable[Path]) -> list[Path]:
    """返回规范化、去重且顺序稳定的 sidecar，供调用方先校验路径冲突。

    创建锁可能改写 sidecar；调用方必须在 acquire 前将这些路径与输入、
    成品和 session/manifest 路径一起检查，不能把用户文件当作锁文件。
    """
    unique: dict[str, Path] = {}
    for path in paths:
        output = path.expanduser().resolve()
        lock_path = Path(str(output) + OUTPUT_LOCK_SUFFIX).resolve()
        unique.setdefault(os.path.normcase(str(lock_path)), lock_path)
    return [unique[key] for key in sorted(unique)]


def acquire_output_locks(
    stack: ExitStack, paths: Iterable[Path], dry_run: bool
) -> None:
    """按固定顺序持锁到 stack 退出；dry-run 不创建目录或文件。"""
    if dry_run:
        return
    for path in output_lock_paths(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stack.enter_context(FileLock(str(path), timeout=30))
        except LockTimeout as exc:
            raise ValueError(f"图片输出正由另一个任务使用，请稍后重试：{path}") from exc
