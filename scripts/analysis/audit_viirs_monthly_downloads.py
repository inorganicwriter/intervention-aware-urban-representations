"""Audit a downloaded monthly VIIRS batch without loading every CSV body."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "collection"))
sys.path.insert(0, str(ROOT / "src"))

from process_viirs_monthly import (  # noqa: E402
    EXPECTED_END,
    EXPECTED_START,
    discover_exports,
    validate_complete_batch,
)

from urban_intervention.data.paths import OUTPUT_DATA_QUALITY_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DATA_QUALITY_DIR / "viirs_monthly_download_audit.json",
    )
    args = parser.parse_args()

    exports = discover_exports(args.input_dir)
    validate_complete_batch(exports)
    sizes = {export.path: export.path.stat().st_size for export in exports}
    cities = sorted({export.city_key for export in exports})
    city_medians = {
        city: statistics.median(sizes[export.path] for export in exports if export.city_key == city)
        for city in cities
    }
    low_size = sorted(
        (
            {
                "city_key": export.city_key,
                "period": export.period,
                "file": export.path.name,
                "bytes": sizes[export.path],
                "city_median_bytes": int(city_medians[export.city_key]),
                "median_ratio": sizes[export.path] / city_medians[export.city_key],
            }
            for export in exports
            if sizes[export.path] < 0.25 * city_medians[export.city_key]
        ),
        key=lambda row: row["median_ratio"],
    )
    sample_paths = {
        next(export.path for export in exports if export.city_key == city) for city in cities
    }
    sample_paths.update(
        export.path
        for export in exports
        if sizes[export.path] < 0.25 * city_medians[export.city_key]
    )
    header_counts: dict[str, int] = {}
    for path in sample_paths:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            header = handle.readline().strip()
        header_counts[header] = header_counts.get(header, 0) + 1

    per_city = []
    for city in cities:
        city_exports = [export for export in exports if export.city_key == city]
        city_sizes = [sizes[export.path] for export in city_exports]
        per_city.append(
            {
                "city_key": city,
                "files": len(city_exports),
                "first_period": min(export.period for export in city_exports),
                "last_period": max(export.period for export in city_exports),
                "min_bytes": min(city_sizes),
                "median_bytes": int(statistics.median(city_sizes)),
                "max_bytes": max(city_sizes),
            }
        )

    report = {
        "schema": "viirs_monthly_download_audit_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "manifest_complete": True,
        "cities": len(cities),
        "files": len(exports),
        "expected_start": EXPECTED_START,
        "expected_end": EXPECTED_END,
        "duplicate_city_periods": len(exports)
        - len({(export.city_key, export.period) for export in exports}),
        "zero_byte_files": sum(size == 0 for size in sizes.values()),
        "total_bytes": sum(sizes.values()),
        "sampled_header_counts": header_counts,
        "low_size_threshold": "below 25% of the same-city median file size",
        "low_size_files": low_size,
        "interpretation": (
            "Low file size is a coverage warning, not proof of corruption. "
            "Grid-month processing must preserve observed-point counts and missing grids."
        ),
        "per_city": per_city,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"audit={args.output}")
    print(f"files={len(exports)} cities={len(cities)} low_size={len(low_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
