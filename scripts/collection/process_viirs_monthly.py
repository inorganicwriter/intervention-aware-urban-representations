"""Compress monthly VIIRS CSV exports into resumable grid-month partitions.

Raw GEE CSV files are read one city-month at a time. Coordinates and repeated
provenance strings are used for validation and spatial matching but are not
copied into the analytical Parquet layer. Hive partition paths carry city,
year, and month, while each Parquet file stores only ``grid_id``, radiance,
valid-day support, and the number of unique source points.

Example::

    python scripts/collection/process_viirs_monthly.py \
        --input-dir <external-monthly-VIIRS-directory> --require-complete
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from urban_intervention.config.project import ACTIVE_CITIES, CITIES, GRID_DIR  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_VIIRS_PARTITION_AUDITS_DIR,
    VIIRS_MONTHLY_DIR,
)

OUT_DIR = VIIRS_MONTHLY_DIR
AUDIT_DIR = OUTPUT_VIIRS_PARTITION_AUDITS_DIR
MAX_MATCH_DISTANCE_M = 500.0
GRID_CELL_SIZE_M = 500.0
GRID_ORIGIN_TOLERANCE_M = 0.05
EXPECTED_START = "2012-01"
EXPECTED_END = "2024-12"
SOURCE_PRODUCT = "NASA/VIIRS/002/VNP46A2"
RAW_COLUMNS = [
    "city",
    "latitude",
    "longitude",
    "month",
    "period",
    "product",
    "radiance",
    "valid_days",
    "year",
]
OUTPUT_COLUMNS = [
    "city_key",
    "grid_id",
    "year",
    "month",
    "avg_rad",
    "valid_days_mean",
    "source_point_count",
]
STORED_COLUMNS = [
    "grid_id",
    "avg_rad",
    "valid_days_mean",
    "source_point_count",
]

_FILE_RE = re.compile(
    r"^viirs_(?P<city>[a-z0-9_]+)_(?P<year>\d{4})_(?P<month>\d{2})"
    r"(?:[-_].*)?$"
)


@dataclass(frozen=True)
class ExportFile:
    path: Path
    city_key: str
    year: int
    month: int

    @property
    def period(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


@dataclass(frozen=True)
class GridLattice:
    """Exact half-open 500 m lattice underlying a clipped city grid."""

    transformer: Transformer
    origin_x: float
    origin_y: float
    lookup: pd.Series
    cell_size_m: float = GRID_CELL_SIZE_M


def transform_arrays(
    transformer: Transformer, x_values: np.ndarray, y_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform arrays without pyproj's deprecated one-element coercion path."""

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if len(x_values) == 1:
        x, y = transformer.transform(float(x_values[0]), float(y_values[0]))
        return np.asarray([x], dtype=float), np.asarray([y], dtype=float)
    x, y = transformer.transform(x_values, y_values)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def parse_export_file(path: Path) -> ExportFile:
    match = _FILE_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unrecognized VIIRS export filename: {path.name}")
    city_key = match.group("city")
    year = int(match.group("year"))
    month = int(match.group("month"))
    if city_key not in CITIES:
        raise ValueError(f"{path.name}: unknown project city {city_key!r}")
    if not 1 <= month <= 12:
        raise ValueError(f"{path.name}: invalid month {month}")
    return ExportFile(path, city_key, year, month)


def discover_exports(input_dir: Path) -> list[ExportFile]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    paths = sorted(input_dir.rglob("viirs_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No viirs_*.csv files found under {input_dir}")
    return [parse_export_file(path) for path in paths]


def expected_periods(start: str = EXPECTED_START, end: str = EXPECTED_END) -> set[str]:
    return {str(period) for period in pd.period_range(start, end, freq="M")}


def validate_complete_batch(exports: list[ExportFile]) -> None:
    expected_cities = set(ACTIVE_CITIES)
    actual_cities = {export.city_key for export in exports}
    missing_cities = sorted(expected_cities - actual_cities)
    extra_cities = sorted(actual_cities - expected_cities)
    if missing_cities or extra_cities:
        raise ValueError(
            "VIIRS city manifest mismatch: "
            f"missing={missing_cities or 'none'}, extra={extra_cities or 'none'}"
        )

    expected = expected_periods()
    failures: list[str] = []
    for city_key in sorted(expected_cities):
        city_exports = [export for export in exports if export.city_key == city_key]
        counts: dict[str, int] = {}
        for export in city_exports:
            counts[export.period] = counts.get(export.period, 0) + 1
        missing = sorted(expected - set(counts))
        extra = sorted(set(counts) - expected)
        duplicates = sorted(period for period, count in counts.items() if count > 1)
        if missing or extra or duplicates:
            failures.append(
                f"{city_key}: missing={missing or 'none'}, extra={extra or 'none'}, "
                f"duplicates={duplicates or 'none'}"
            )
    if failures:
        raise ValueError("Incomplete VIIRS month manifest:\n" + "\n".join(failures))


def load_grids(city_key: str) -> pd.DataFrame:
    path = GRID_DIR / city_key / f"{city_key}_grids.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Grid file not found: {path}")
    grids = pd.read_parquet(path, columns=["grid_id", "row", "col", "centroid_lon", "centroid_lat"])
    required = ["grid_id", "row", "col", "centroid_lon", "centroid_lat"]
    if grids[required].isna().any(axis=None):
        raise ValueError(f"{city_key}: null reference-grid keys or centroids")
    if grids["grid_id"].duplicated().any():
        raise ValueError(f"{city_key}: duplicate reference-grid keys")
    if grids[["row", "col"]].duplicated().any():
        raise ValueError(f"{city_key}: duplicate reference-grid row/column keys")
    return grids


def read_export(export: ExportFile) -> pd.DataFrame:
    """Read only fields used for validation, matching, and the final outcome.

    Uses pyarrow's CSV reader for ~13x speedup over pandas.read_csv, with
    column types enforced at parse time so downstream code can skip
    per-row ``pd.to_numeric`` coercion.
    """

    convert_options = pa_csv.ConvertOptions(
        column_types={
            "city": pa.string(),
            "latitude": pa.float64(),
            "longitude": pa.float64(),
            "month": pa.int8(),
            "period": pa.string(),
            "product": pa.string(),
            "radiance": pa.float32(),
            "valid_days": pa.float32(),
            "year": pa.int16(),
        },
        include_columns=RAW_COLUMNS,
    )
    try:
        table = pa_csv.read_csv(export.path, convert_options=convert_options)
    except (pa.ArrowInvalid, pa.ArrowTypeError, OSError, KeyError) as error:
        raise ValueError(f"{export.path.name}: incompatible CSV schema: {error}") from error
    return normalize_export(table.to_pandas(), export)


def normalize_export(frame: pd.DataFrame, export: ExportFile) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    aliases = {"longitude": "lon", "latitude": "lat", "radiance": "avg_rad"}
    for source, target in aliases.items():
        if source in frame.columns and target not in frame.columns:
            frame = frame.rename(columns={source: target})

    required = {"year", "month", "avg_rad", "valid_days", "lon", "lat"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{export.path.name}: missing columns {sorted(missing)}")

    if "city" in frame.columns:
        cities = frame["city"].dropna().unique()
        if len(cities) and {str(c).strip() for c in cities} != {export.city_key}:
            raise ValueError(f"{export.path.name}: filename city differs from rows {set(cities)}")
    if "product" in frame.columns:
        products = frame["product"].dropna().unique()
        if len(products) and {str(p).strip() for p in products} != {SOURCE_PRODUCT}:
            raise ValueError(f"{export.path.name}: unexpected products {set(products)}")

    # Column types are enforced by the pyarrow CSV reader at parse time, so
    # per-row pd.to_numeric coercion is no longer needed for the common path.
    # When normalize_export is called directly with an untyped frame (e.g. an
    # empty DataFrame in tests), ensure numeric dtypes so downstream vectorized
    # operations like np.isfinite do not fail on object columns.
    for column in ["avg_rad", "valid_days", "lon", "lat"]:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[["year", "month"]].isna().any(axis=None):
        raise ValueError(f"{export.path.name}: null or nonnumeric year/month")
    if not ((frame["year"] == export.year) & (frame["month"] == export.month)).all():
        raise ValueError(f"{export.path.name}: row period differs from filename")
    invalid_days = frame["valid_days"].notna() & ~frame["valid_days"].between(0, 31)
    if invalid_days.any():
        raise ValueError(f"{export.path.name}: valid_days outside [0, 31]")
    if "period" in frame.columns:
        labels = frame["period"].dropna().str[:7].unique()
        if len(labels) and set(labels) != {export.period}:
            raise ValueError(f"{export.path.name}: period column differs from filename")

    frame["city_key"] = export.city_key
    frame["year"] = np.int16(export.year)
    frame["month"] = np.int8(export.month)
    return frame[["city_key", "year", "month", "avg_rad", "valid_days", "lon", "lat"]]


def build_grid_lattice(
    grids: pd.DataFrame,
    projected_crs: str,
    cell_size_m: float = GRID_CELL_SIZE_M,
) -> GridLattice:
    """Recover the exact projected-grid origin from persisted row/column keys."""

    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    centroid_x, centroid_y = transform_arrays(
        transformer,
        grids["centroid_lon"].to_numpy(dtype=float),
        grids["centroid_lat"].to_numpy(dtype=float),
    )
    origin_x_values = centroid_x - (
        grids["col"].to_numpy(dtype=float) * cell_size_m + cell_size_m / 2
    )
    origin_y_values = centroid_y - (
        grids["row"].to_numpy(dtype=float) * cell_size_m + cell_size_m / 2
    )
    origin_x = float(np.median(origin_x_values))
    origin_y = float(np.median(origin_y_values))
    maximum_origin_residual = max(
        float(np.max(np.abs(origin_x_values - origin_x))),
        float(np.max(np.abs(origin_y_values - origin_y))),
    )
    if maximum_origin_residual > GRID_ORIGIN_TOLERANCE_M:
        raise ValueError(
            "Reference grids are not a stable projected 500 m lattice: "
            f"maximum origin residual={maximum_origin_residual:.4f} m"
        )
    lookup = pd.Series(
        grids["grid_id"].astype(str).to_numpy(),
        index=pd.MultiIndex.from_arrays(
            [grids["row"].astype("int32"), grids["col"].astype("int32")],
            names=["row", "col"],
        ),
        dtype="string",
    )
    return GridLattice(transformer, origin_x, origin_y, lookup, cell_size_m)


def assign_grid_exact(
    points: pd.DataFrame,
    lattice: GridLattice,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assign points by exact projected cell containment, never nearest-centroid."""

    coordinate_valid = (
        np.isfinite(points["lon"])
        & np.isfinite(points["lat"])
        & points["lon"].between(-180, 180)
        & points["lat"].between(-90, 90)
    )
    valid = points.loc[coordinate_valid].copy()
    diagnostics = {
        "source_rows": int(len(points)),
        "invalid_coordinate_rows": int((~coordinate_valid).sum()),
        "outside_reference_grid_rows": 0,
        "matched_rows": 0,
    }
    if valid.empty:
        valid["grid_id"] = pd.Series(dtype="string")
        valid["cell_edge_distance_m"] = pd.Series(dtype="float32")
        return valid, diagnostics

    point_x, point_y = transform_arrays(
        lattice.transformer, valid["lon"].to_numpy(dtype=float), valid["lat"].to_numpy(dtype=float)
    )
    cols = np.floor((point_x - lattice.origin_x) / lattice.cell_size_m).astype("int32")
    rows = np.floor((point_y - lattice.origin_y) / lattice.cell_size_m).astype("int32")
    cell_keys = pd.MultiIndex.from_arrays([rows, cols], names=["row", "col"])
    assigned = lattice.lookup.reindex(cell_keys).to_numpy()
    matched = pd.notna(assigned)
    diagnostics["outside_reference_grid_rows"] = int((~matched).sum())
    diagnostics["matched_rows"] = int(matched.sum())

    result = valid.loc[matched].copy()
    result["grid_id"] = assigned[matched]
    local_x = (point_x[matched] - lattice.origin_x) % lattice.cell_size_m
    local_y = (point_y[matched] - lattice.origin_y) % lattice.cell_size_m
    result["cell_edge_distance_m"] = np.minimum.reduce(
        [
            local_x,
            lattice.cell_size_m - local_x,
            local_y,
            lattice.cell_size_m - local_y,
        ]
    ).astype("float32")
    return result, diagnostics


def canonicalize_source_coordinates(
    points: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse repeat exports of one VIIRS coordinate, rejecting conflicts.

    A source coordinate is one raster sample. Repeated copies of the same
    coordinate in a city-month must therefore agree on radiance and valid-day
    support. Conflicting values indicate mixed or corrupted exports and are not
    silently averaged.
    """

    coordinate_key = ["city_key", "year", "month", "lon", "lat"]
    if points.empty:
        return points.copy(), {
            "duplicate_coordinate_rows_removed": 0,
            "conflicting_coordinate_groups": 0,
            "unique_source_coordinates": 0,
        }
    duplicate_mask = points.duplicated(coordinate_key, keep=False)
    conflicting_groups = 0
    if duplicate_mask.any():
        repeated = points.loc[duplicate_mask]
        conflicts = repeated.groupby(coordinate_key, observed=True, dropna=False).agg(
            radiance_values=("avg_rad", lambda values: values.nunique(dropna=False)),
            valid_day_values=("valid_days", lambda values: values.nunique(dropna=False)),
        )
        conflicting_groups = int(
            (conflicts["radiance_values"].gt(1) | conflicts["valid_day_values"].gt(1)).sum()
        )
    if conflicting_groups:
        raise ValueError(
            f"VIIRS contains {conflicting_groups} same-coordinate groups with conflicting values"
        )
    canonical = points.drop_duplicates(coordinate_key, keep="first").copy()
    return canonical, {
        "duplicate_coordinate_rows_removed": int(len(points) - len(canonical)),
        "conflicting_coordinate_groups": conflicting_groups,
        "unique_source_coordinates": int(len(canonical)),
    }


def aggregate_grid_month(points: pd.DataFrame, already_canonicalized: bool = False) -> pd.DataFrame:
    """Average unique equal-scale VIIRS samples within each target grid.

    When ``already_canonicalized`` is True the caller guarantees that
    ``canonicalize_source_coordinates`` has already run; this skips a
    redundant full-table dedupe pass.
    """

    if points.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not already_canonicalized:
        points, _ = canonicalize_source_coordinates(points)
    points = points[np.isfinite(points["avg_rad"])].copy()
    if points.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    keys = ["city_key", "grid_id", "year", "month"]
    result = points.groupby(keys, as_index=False, observed=True).agg(
        avg_rad=("avg_rad", "mean"),
        valid_days_mean=("valid_days", "mean"),
        source_point_count=("avg_rad", "size"),
    )
    result["city_key"] = result["city_key"].astype("string")
    result["grid_id"] = result["grid_id"].astype("string")
    result["year"] = result["year"].astype("int16")
    result["month"] = result["month"].astype("int8")
    result["avg_rad"] = result["avg_rad"].astype("float32")
    result["valid_days_mean"] = result["valid_days_mean"].astype("float32")
    result["source_point_count"] = result["source_point_count"].astype("uint16")
    if result.duplicated(keys).any():
        raise AssertionError("VIIRS grid-month primary-key contract failed")
    return result[OUTPUT_COLUMNS].sort_values(keys, kind="stable").reset_index(drop=True)


def partition_path(output_dir: Path, export: ExportFile) -> Path:
    return (
        output_dir
        / f"city_key={export.city_key}"
        / f"year={export.year:04d}"
        / f"month={export.month:02d}"
        / "part.parquet"
    )


def audit_path(audit_dir: Path, export: ExportFile) -> Path:
    return audit_dir / export.city_key / f"{export.year:04d}-{export.month:02d}.json"


def atomic_parquet(frame: pd.DataFrame, path: Path, compression_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=compression_level,
        use_dictionary=True,
        write_statistics=True,
    )
    temporary.replace(path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_period(
    export: ExportFile,
    lattice: GridLattice,
    output_dir: Path,
    audit_dir: Path,
    compression_level: int = 9,
    force: bool = False,
) -> tuple[Path, dict[str, object]]:
    output = partition_path(output_dir, export)
    audit_output = audit_path(audit_dir, export)
    if output.exists() and audit_output.exists() and not force:
        return output, {"status": "skipped", "city_key": export.city_key, "period": export.period}

    frame = read_export(export)
    matched, diagnostics = assign_grid_exact(frame, lattice)
    canonical_points, coordinate_audit = canonicalize_source_coordinates(matched)
    invalid_radiance = int((~np.isfinite(canonical_points["avg_rad"])).sum())
    edge_distance_min = float(matched["cell_edge_distance_m"].min()) if len(matched) else None
    near_boundary_rows = int(matched["cell_edge_distance_m"].lt(1.0).sum())
    finite_points = canonical_points[np.isfinite(canonical_points["avg_rad"])].copy()
    grid_counts = finite_points.groupby("grid_id", observed=True).size()
    multi_point_grids = int(grid_counts.gt(1).sum())
    compact = aggregate_grid_month(canonical_points, already_canonicalized=True)
    stored = compact[STORED_COLUMNS]
    atomic_parquet(stored, output, compression_level)
    input_bytes = export.path.stat().st_size
    output_bytes = output.stat().st_size
    audit = {
        "schema": "viirs_monthly_partition_audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "written",
        "city_key": export.city_key,
        "year": export.year,
        "month": export.month,
        "period": export.period,
        "input_file": export.path.name,
        "input_bytes": input_bytes,
        **diagnostics,
        **coordinate_audit,
        "invalid_radiance_rows_removed": invalid_radiance,
        "reference_grid_rows": int(len(lattice.lookup)),
        "output_grid_rows": int(len(stored)),
        "grid_coverage_share": len(stored) / len(lattice.lookup) if len(lattice.lookup) else None,
        "multi_source_point_grids": multi_point_grids,
        "multi_source_point_grid_share": multi_point_grids / len(stored) if len(stored) else None,
        "maximum_source_points_per_grid": int(grid_counts.max()) if len(grid_counts) else 0,
        "within_grid_aggregation": "unweighted_mean_of_unique_source_coordinates",
        "assignment_method": "exact_projected_lattice_half_open_cells",
        "cell_edge_distance_m_min": edge_distance_min,
        "rows_within_1m_of_cell_edge": near_boundary_rows,
        "output_bytes": output_bytes,
        "csv_to_parquet_ratio": output_bytes / input_bytes if input_bytes else None,
        "stored_columns": STORED_COLUMNS,
        "compression": f"zstd_level_{compression_level}",
    }
    atomic_json(audit, audit_output)
    return output, audit


def process_directory(
    input_dir: Path,
    require_complete: bool = False,
    city: str | None = None,
    period: str | None = None,
    force: bool = False,
    output_dir: Path | None = None,
    audit_dir: Path | None = None,
    compression_level: int = 9,
) -> list[Path]:
    output_dir = output_dir or OUT_DIR
    audit_dir = audit_dir or AUDIT_DIR
    exports = discover_exports(input_dir)
    if require_complete:
        validate_complete_batch(exports)
    if city is not None:
        if city not in ACTIVE_CITIES:
            raise ValueError(f"Unknown active city: {city}")
        exports = [export for export in exports if export.city_key == city]
    if period is not None:
        if not re.fullmatch(r"\d{4}-\d{2}", period):
            raise ValueError("period must use YYYY-MM")
        exports = [export for export in exports if export.period == period]
    if not exports:
        raise ValueError("No VIIRS exports remain after filters")

    outputs: list[Path] = []
    by_city: dict[str, list[ExportFile]] = {}
    for export in exports:
        by_city.setdefault(export.city_key, []).append(export)
    print(f"Found {len(exports):,} partitions across {len(by_city)} cities")
    for city_key, city_exports in sorted(by_city.items()):
        grids = load_grids(city_key)
        lattice = build_grid_lattice(grids, str(CITIES[city_key]["projected_crs"]))
        for export in sorted(
            city_exports, key=lambda item: (item.year, item.month, item.path.name)
        ):
            try:
                output, audit = process_period(
                    export,
                    lattice,
                    output_dir,
                    audit_dir,
                    compression_level=compression_level,
                    force=force,
                )
                outputs.append(output)
                print(
                    f"[{export.city_key} {export.period}] {audit['status']} "
                    f"rows={audit.get('output_grid_rows', '-')} -> {output.relative_to(output_dir)}"
                )
            except Exception as error:
                failure_audit = {
                    "schema": "viirs_monthly_partition_audit",
                    "generated_at_utc": datetime.now(UTC).isoformat(),
                    "status": "error",
                    "city_key": export.city_key,
                    "year": export.year,
                    "month": export.month,
                    "period": export.period,
                    "input_file": export.path.name,
                    "error": str(error),
                }
                atomic_json(failure_audit, audit_path(audit_dir, export))
                print(f"[{export.city_key} {export.period}] error: {error}")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--city", choices=ACTIVE_CITIES)
    parser.add_argument("--period", help="Optional single month in YYYY-MM format")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--compression-level", type=int, default=9, choices=range(1, 23))
    args = parser.parse_args()
    process_directory(
        args.input_dir,
        require_complete=args.require_complete,
        city=args.city,
        period=args.period,
        force=args.force,
        output_dir=args.output_dir,
        audit_dir=args.audit_dir,
        compression_level=args.compression_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
