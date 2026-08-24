"""Shared utilities for the urban-intervention package."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file with the package-wide streaming implementation."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: Any, columns: Iterable[str], label: str) -> None:
    """Require named columns on a dataframe-like object."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f"{path.name}.{os.getpid()}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def _replace_with_retry(temporary: Path, path: Path, attempts: int) -> None:
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def atomic_write_csv(
    frame: Any,
    path: str | Path,
    *,
    encoding: str = "utf-8-sig",
    permission_attempts: int = 5,
) -> None:
    """Atomically publish a CSV without exposing a partially written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        frame.to_csv(temporary, index=False, encoding=encoding)
        _replace_with_retry(temporary, target, permission_attempts)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(frame: Any, path: str | Path) -> None:
    """Atomically publish a zstd-compressed Parquet dataframe."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        _replace_with_retry(temporary, target, 5)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    payload: Any,
    path: str | Path,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Atomically publish UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=ensure_ascii,
                indent=indent,
                default=default,
            ),
            encoding="utf-8",
        )
        _replace_with_retry(temporary, target, 5)
    finally:
        temporary.unlink(missing_ok=True)
