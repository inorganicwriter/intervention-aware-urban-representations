"""Materialize only the VIIRS city-month partitions needed by causal jobs.

This is a cache coordinator around ``process_viirs_monthly.py``. It discovers
the immutable raw exports once, loads a city's grid lattice once, and processes
only missing requested months. Existing Parquet+audit pairs are left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import process_viirs_monthly as processor

from urban_intervention.data.paths import OUTPUT_VIIRS_MONTHLY_DIR

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = (
    Path(os.environ["MIT_VIIRS_RAW"]).expanduser() if os.environ.get("MIT_VIIRS_RAW") else None
)
DEFAULT_MANIFEST = OUTPUT_VIIRS_MONTHLY_DIR / "jit_cache_manifest.json"


def requested_periods(start: str, end: str) -> list[str]:
    periods = pd.period_range(start, end, freq="M")
    if not len(periods):
        raise ValueError("VIIRS cache request has no months")
    return [str(period) for period in periods]


def ensure_partitions(
    input_dir: Path | None,
    city_key: str,
    periods: list[str],
    output_dir: Path = processor.OUT_DIR,
    audit_dir: Path = processor.AUDIT_DIR,
    compression_level: int = 9,
) -> dict[str, object]:
    if city_key not in processor.ACTIVE_CITIES:
        raise ValueError(f"Unknown active city: {city_key}")
    periods = sorted(set(periods))
    invalid = [period for period in periods if len(period) != 7 or period[4] != "-"]
    if invalid:
        raise ValueError(f"Invalid YYYY-MM periods: {invalid}")

    requested = {(city_key, int(period[:4]), int(period[5:7])) for period in periods}
    cached: list[str] = []
    needed: set[tuple[str, int, int]] = set()
    for key in requested:
        export = processor.ExportFile(Path("unused.csv"), key[0], key[1], key[2])
        parquet = processor.partition_path(output_dir, export)
        audit = processor.audit_path(audit_dir, export)
        if parquet.exists() and audit.exists():
            cached.append(export.period)
        else:
            needed.add(key)

    processed: list[str] = []
    if needed:
        if input_dir is None:
            raise ValueError("Missing uncached VIIRS months; pass --input-dir or set MIT_VIIRS_RAW")
        exports = {
            (item.city_key, item.year, item.month): item
            for item in processor.discover_exports(input_dir)
            if (item.city_key, item.year, item.month) in needed
        }
        missing_raw = sorted(needed - set(exports))
        if missing_raw:
            raise FileNotFoundError(f"Missing raw VIIRS exports: {missing_raw}")
        grids = processor.load_grids(city_key)
        lattice = processor.build_grid_lattice(
            grids, str(processor.CITIES[city_key]["projected_crs"])
        )
        for key in sorted(needed):
            export = exports[key]
            processor.process_period(
                export,
                lattice,
                output_dir,
                audit_dir,
                compression_level=compression_level,
                force=False,
            )
            processed.append(export.period)

    return {
        "schema": "viirs_monthly_jit_cache_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "city_key": city_key,
        "requested_periods": periods,
        "already_cached": sorted(cached),
        "processed": processed,
        "raw_input_dir": str(input_dir.resolve()) if input_dir is not None else None,
        "output_dir": str(output_dir.resolve()),
    }


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--city", required=True, choices=processor.ACTIVE_CITIES)
    parser.add_argument("--start", required=True, help="First month, YYYY-MM")
    parser.add_argument("--end", required=True, help="Last month, YYYY-MM")
    parser.add_argument("--output-dir", type=Path, default=processor.OUT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=processor.AUDIT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--compression-level", type=int, default=9, choices=range(1, 23))
    args = parser.parse_args()
    manifest = ensure_partitions(
        args.input_dir,
        args.city,
        requested_periods(args.start, args.end),
        output_dir=args.output_dir,
        audit_dir=args.audit_dir,
        compression_level=args.compression_level,
    )
    atomic_json(manifest, args.manifest)
    print(
        f"VIIRS JIT cache {args.city}: requested={len(manifest['requested_periods'])} "
        f"cached={len(manifest['already_cached'])} processed={len(manifest['processed'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
