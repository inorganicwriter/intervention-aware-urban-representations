"""Atomically filter a generated city POI panel to a target year range."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd


def _temporary_path(destination: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = _temporary_path(destination)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def filter_years(
    path: Path,
    start_year: int,
    end_year: int,
    backup_path: Path | None = None,
) -> tuple[int, Path]:
    path = path.resolve()
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    if not path.is_file():
        raise FileNotFoundError(path)
    backup = (
        backup_path or path.with_name(f"{path.stem}.pre_filter_backup{path.suffix}")
    ).resolve()
    if backup == path:
        raise ValueError("Backup path must differ from the input path")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")

    frame = pd.read_parquet(path)
    if "year" not in frame:
        raise ValueError("POI panel lacks required 'year' column")
    years = pd.to_numeric(frame["year"], errors="coerce")
    output = frame.loc[years.between(start_year, end_year)].copy()

    temporary = _temporary_path(path)
    try:
        output.to_parquet(temporary, index=False)
        verified = pd.read_parquet(temporary)
        if len(verified) != len(output) or list(verified.columns) != list(output.columns):
            raise OSError("Temporary POI panel failed validation")
        backup.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copy(path, backup)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(output), backup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args(argv)
    row_count, backup = filter_years(
        args.path, args.start_year, args.end_year, backup_path=args.backup
    )
    print(
        f"Saved {args.path} with {row_count:,} rows for "
        f"{args.start_year}-{args.end_year}; backup={backup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
