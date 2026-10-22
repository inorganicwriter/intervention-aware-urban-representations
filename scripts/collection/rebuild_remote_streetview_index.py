"""Read raw collector assets and bind actual panorama positions to frozen grids.

Standalone on the asset server: needs numpy, pandas, Shapely 2 and Pillow.
It writes only a new output directory and never edits the collector's archive.
CSV transport avoids changing the server's installed dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import shapely
from PIL import Image

try:
    from .streetview_coordinates import actual_pano_coordinates
except ImportError:
    from streetview_coordinates import actual_pano_coordinates

FIELDS = ["province", "city", "county", "wgs_lon", "wgs_lat", "calc_bd_x", "calc_bd_y",
          "panoid", "ret_bd_x", "ret_bd_y", "capture_date", "status"]
INDEX_FIELDS = ["city_key", "grid_id", "grid_geometry_sha256", "panoid", "capture_date",
                "capture_month", "capture_date_precision", "wgs_lon", "wgs_lat",
                "query_wgs_lon", "query_wgs_lat", "ret_bd_x", "ret_bd_y", "mc_divisor",
                "asset_path", "asset_bytes", "source_metadata"]
RECIPE = "actual_pano_scaled_mc_month_cutoff_exact_asset_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_unique(lon: np.ndarray, lat: np.ndarray, grids: list[dict]) -> np.ndarray:
    polygons = shapely.from_wkt([g["geometry_wkt"] for g in grids])
    points = shapely.points(lon, lat)
    point_i, grid_i = shapely.STRtree(polygons).query(points, predicate="covered_by")
    counts = np.bincount(point_i, minlength=len(points))
    unique = counts[point_i] == 1
    result = np.full(len(points), -1, dtype=int)
    result[point_i[unique]] = grid_i[unique]
    return result


def verify_image(path: str) -> dict:
    """Fully decode selected training assets and hash their original bytes."""
    try:
        source = Path(path)
        before = source.stat()
        with Image.open(source) as image:
            image.load()
            width, height = image.size
            mode, format_name = image.mode, image.format
        digest = sha256(source)
        after = source.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ValueError("asset changed during verification")
        if width < 128 or height < 64:
            raise ValueError("image too small")
        return {"asset_path": path, "image_valid": True, "width": width, "height": height,
                "image_mode": mode, "image_format": format_name, "asset_sha256": digest,
                "asset_bytes": after.st_size, "error": ""}
    except Exception as exc:
        return {"asset_path": path, "image_valid": False, "width": 0, "height": 0,
                "image_mode": "", "image_format": "", "asset_sha256": "",
                "asset_bytes": 0, "error": f"{type(exc).__name__}: {exc}"}


def build_city(city: str, spec: dict, root: Path, output: Path, contract_sha: str,
               metadata_only: bool = False) -> tuple[pd.DataFrame, dict]:
    meta_dir = root / "metadata/success" / spec["province"] / spec["source_city"]
    files = sorted(meta_dir.glob("*.csv"))
    audit = {"contract_sha256": contract_sha, "recipe": RECIPE, "metadata_only_index": metadata_only,
             "county_null_policy": "candidate_source_city_then_verify_actual_directory",
             "metadata_files": [], "raw_rows": 0, "success_rows": 0,
             "invalid_actual_coordinate_rows": 0, "invalid_date_rows": 0,
             "fixed_100_unit_rows": 0, "matched_query_grid_rows": 0,
             "matched_actual_grid_rows_before_dedup": 0, "missing_image_panos": 0,
             "conflicting_panos": 0, "rows_with_image": 0}
    if not files:
        audit["status"] = "metadata_missing"
        return pd.DataFrame(columns=INDEX_FIELDS), audit
    parts = []
    for path in files:
        before = path.stat()
        for frame in pd.read_csv(path, dtype=str, usecols=FIELDS, chunksize=100000,
                                 encoding="utf-8-sig"):
            audit["raw_rows"] += len(frame)
            frame = frame.loc[frame.status.str.strip().eq("成功")].copy()
            audit["success_rows"] += len(frame)
            for name in ["wgs_lon", "wgs_lat", "calc_bd_x", "calc_bd_y", "ret_bd_x", "ret_bd_y"]:
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
            lon, lat, scale = actual_pano_coordinates(
                *(frame[k].to_numpy() for k in ["ret_bd_x", "ret_bd_y", "calc_bd_x", "calc_bd_y"]))
            valid = np.isfinite(lon) & np.isfinite(lat)
            audit["invalid_actual_coordinate_rows"] += int((~valid).sum())
            audit["fixed_100_unit_rows"] += int((scale == 100).sum())
            frame["query_wgs_lon"], frame["query_wgs_lat"] = frame.wgs_lon, frame.wgs_lat
            frame["wgs_lon"], frame["wgs_lat"], frame["mc_divisor"] = lon, lat, scale
            frame = frame.loc[valid].copy()
            parsed = pd.to_datetime(frame.capture_date, errors="coerce")
            # The collector parse_baidu_date records only year/month. Do not treat day=1 as known.
            bad_date = parsed.isna() | parsed.gt(pd.Timestamp.now()) | parsed.lt("2000-01-01")
            audit["invalid_date_rows"] += int(bad_date.sum())
            frame = frame.loc[~bad_date].copy()
            frame["capture_month"] = parsed.loc[~bad_date].dt.strftime("%Y-%m")
            frame["capture_date"] = frame.capture_month + "-01"
            frame["capture_date_precision"] = "month"
            frame["county"] = frame.county.fillna(spec["source_city"])
            frame = frame.dropna(subset=["panoid"]).drop_duplicates(
                ["panoid", "ret_bd_x", "ret_bd_y", "capture_month"])
            assigned = assign_unique(frame.wgs_lon.to_numpy(), frame.wgs_lat.to_numpy(), spec["grids"])
            query_assigned = assign_unique(frame.query_wgs_lon.to_numpy(), frame.query_wgs_lat.to_numpy(), spec["grids"])
            audit["matched_query_grid_rows"] += int((query_assigned >= 0).sum())
            frame = frame.loc[assigned >= 0].copy()
            assigned = assigned[assigned >= 0]
            audit["matched_actual_grid_rows_before_dedup"] += len(frame)
            if frame.empty:
                continue
            frame["grid_id"] = [spec["grids"][i]["grid_id"] for i in assigned]
            frame["grid_geometry_sha256"] = [spec["grids"][i]["grid_geometry_sha256"] for i in assigned]
            frame["source_metadata"] = str(path)
            parts.append(frame)
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source metadata changed: {path}")
        audit["metadata_files"].append({"path": str(path), "bytes": after.st_size,
                                        "mtime_ns": after.st_mtime_ns, "sha256": digest})
        print(json.dumps({"city": city, "file": path.name, "processed_rows": audit["raw_rows"]}), flush=True)
    if not parts:
        audit["status"] = "no_captures_in_treated_grids"
        return pd.DataFrame(columns=INDEX_FIELDS), audit
    frame = pd.concat(parts, ignore_index=True)
    # Different query points can return the same panorama; keep its physical position once.
    distinct = frame.groupby("panoid")[["ret_bd_x", "ret_bd_y", "capture_month", "grid_id"]].nunique()
    conflict_ids = distinct.index[distinct.gt(1).any(axis=1)]
    audit["conflicting_panos"] = len(conflict_ids)
    frame = frame.loc[~frame.panoid.isin(conflict_ids)].drop_duplicates("panoid").copy()
    rows = []
    image_city = root / "images" / spec["province"] / spec["source_city"]
    counties = [] if metadata_only else sorted(p for p in image_city.iterdir() if p.is_dir()) if image_city.is_dir() else []
    for record in frame.to_dict("records"):
        pano = str(record["panoid"])
        # Exact collector filename/layout; do not use a prefix glob which can misidentify a pano.
        path = root / "images" / str(record["province"]) / str(record["city"]) / str(record["county"]) / pano[-2] / pano[-1] / f"{pano}.jpg"
        if metadata_only:
            record.update(city_key=city, asset_path=str(path), asset_bytes=0)
            rows.append({name: record[name] for name in INDEX_FIELDS})
            continue
        if not path.is_file():
            # Collector globally deduplicates download tasks; a neighboring county
            # query may have caused the same pano to be saved in that county instead.
            alternatives = [d / pano[-2] / pano[-1] / f"{pano}.jpg" for d in counties]
            path = next((candidate for candidate in alternatives if candidate.is_file()), path)
        try:
            info = path.stat()
        except FileNotFoundError:
            info = None
        if info is None or not stat.S_ISREG(info.st_mode) or info.st_size == 0:
            audit["missing_image_panos"] += 1
            continue
        record.update(city_key=city, asset_path=str(path), asset_bytes=info.st_size)
        rows.append({name: record[name] for name in INDEX_FIELDS})
    result = pd.DataFrame(rows, columns=INDEX_FIELDS)
    audit.update(status="metadata_indexed" if metadata_only else "indexed",
                 matched_metadata_panos=len(result), rows_with_image=None if metadata_only else len(result),
                 earliest_month=None if result.empty else result.capture_month.min(),
                 latest_month=None if result.empty else result.capture_month.max())
    result.to_csv(output / f"{city}_index.csv", index=False)
    (output / f"{city}_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"city": city, "raw_rows": audit["raw_rows"], "index_rows": len(result)}), flush=True)
    return result, audit


def select_pre_assets(events: pd.DataFrame, index: pd.DataFrame, max_images: int) -> pd.DataFrame:
    event_fields = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]
    matched = events[event_fields].merge(index, on=["city_key", "grid_id"], how="inner")
    matched = matched.loc[matched.capture_month < matched.opening_month.str[:7]].copy()
    return matched.sort_values(["treatment_order", "capture_month", "panoid"],
                               ascending=[True, False, True]).groupby("treatment_order").head(max_images)


def process_city_work(item):
    city, spec, root, output, contract_sha, metadata_only = item
    index_file, audit_file = output / f"{city}_index.csv", output / f"{city}_audit.json"
    if index_file.exists() and audit_file.exists():
        audit = json.loads(audit_file.read_text())
        unchanged = audit.get("recipe") == RECIPE and audit.get("contract_sha256") == contract_sha
        unchanged &= audit.get("metadata_only_index", False) == metadata_only
        if audit.get("county_null_policy") is None:
            unchanged &= not any(Path(s["path"]).stem.casefold() in {"nan", "none", "unknown"} for s in audit["metadata_files"])
        unchanged &= all(Path(s["path"]).stat().st_size == s["bytes"] and
                         Path(s["path"]).stat().st_mtime_ns == s["mtime_ns"] for s in audit["metadata_files"])
        unchanged &= {str(p) for p in (root / "metadata/success" / spec["province"] / spec["source_city"]).glob("*.csv")} == {s["path"] for s in audit["metadata_files"]}
        if unchanged:
            print(json.dumps({"city": city, "resumed": True}), flush=True)
            return city, pd.read_csv(index_file, dtype={"panoid": str, "capture_month": str}), audit
    frame, audit = build_city(city, spec, root, output, contract_sha, metadata_only)
    return city, frame, audit


class ImageDirectoryInventory:
    """Resolve exact filenames via positive directory listings, avoiding slow NFS negative lookups."""

    def __init__(self, root: Path):
        self.root = root
        self.cache = {}
        self.lock = Lock()

    def child_names(self, parent: Path) -> set[str]:
        with self.lock:
            cached = self.cache.get(parent)
        if cached is not None:
            return cached
        with os.scandir(parent) as entries:
            names = {entry.name for entry in entries}
        with self.lock:
            return self.cache.setdefault(parent, names)

    def known_path(self, candidate: Path) -> bool:
        if ".." in candidate.parts or not candidate.is_relative_to(self.root):
            return False
        current = self.root
        try:
            for part in candidate.relative_to(self.root).parts:
                if part not in self.child_names(current):
                    return False
                current = current / part
            return True
        except (OSError, ValueError):
            return False


def select_verified_pre(events: pd.DataFrame, index: pd.DataFrame, contract: dict,
                        root: Path, max_images: int, workers: int):
    matched = events[["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]].merge(
        index, on=["city_key", "grid_id"], how="inner")
    matched = matched.loc[matched.capture_month < matched.opening_month.str[:7]].sort_values(
        ["treatment_order", "capture_month", "panoid"], ascending=[True, False, True])
    return select_verified_candidates(matched, contract, root, max_images, workers, ["treatment_order"])


def select_verified_candidates(matched: pd.DataFrame, contract: dict, root: Path,
                               max_images: int, workers: int, group_keys: list[str]):
    """Verify already gated candidates; temporal roles are set by the caller."""
    inventory = ImageDirectoryInventory(root / "images")

    def checked_image(path):
        if not inventory.known_path(path):
            return {"asset_path": str(path), "image_valid": False, "width": 0, "height": 0,
                    "image_mode": "", "image_format": "", "asset_sha256": "", "asset_bytes": 0,
                    "error": "filename absent from image directory inventory"}
        return verify_image(str(path))

    def choose(group):
        chosen, checked = [], []
        for row in group.to_dict("records"):
            expected = Path(row["asset_path"])
            check = checked_image(expected)
            checked.append(check)
            if not check["image_valid"]:
                spec, pano = contract["cities"][row["city_key"]], row["panoid"]
                base = root / "images" / spec["province"] / spec["source_city"]
                counties = sorted(inventory.child_names(base)) if inventory.known_path(base) else []
                for county in counties:
                    alternate = root / "images" / spec["province"] / spec["source_city"] / county / pano[-2] / pano[-1] / f"{pano}.jpg"
                    if alternate == expected:
                        continue
                    check = checked_image(alternate)
                    checked.append(check)
                    if check["image_valid"]:
                        break
            if check["image_valid"]:
                row.update(check)
                chosen.append(row)
            if len(chosen) >= max_images:
                break
        return chosen, checked

    with ThreadPoolExecutor(workers) as pool:
        groups = [group for _, group in matched.groupby(group_keys)]
        results = []
        for number, result in enumerate(pool.map(choose, groups), 1):
            results.append(result)
            if number % 500 == 0 or number == len(groups):
                print(json.dumps({"verified_candidate_groups": number, "total_candidate_groups": len(groups)}), flush=True)
    pre = pd.DataFrame([row for chosen, _ in results for row in chosen], columns=[*matched.columns,
                       "image_valid", "width", "height", "image_mode", "image_format", "asset_sha256", "error"])
    quality = pd.DataFrame([row for _, checked in results for row in checked], columns=[
        "asset_path", "image_valid", "width", "height", "image_mode", "image_format", "asset_sha256", "asset_bytes", "error"
    ]).drop_duplicates("asset_path")
    return pre, quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--city", default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--city-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--metadata-only-index", action="store_true",
                        help="Verify only selected training images, retaining unverified metadata candidates.")
    parser.add_argument("--verify-existing-index", action="store_true",
                        help="Verify a previously completed local metadata index without rereading NAS source files.")
    args = parser.parse_args()
    if args.max_images < 1:
        parser.error("max-images must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_sha = sha256(args.contract)
    events = pd.DataFrame(contract["events"])
    cities = sorted(contract["cities"]) if args.city == "all" else args.city.split(",")
    if args.verify_existing_index:
        if not args.metadata_only_index:
            parser.error("verify-existing-index requires metadata-only-index")
        audits = {city: json.loads((args.output / f"{city}_audit.json").read_text()) for city in cities}
        if any(audit["contract_sha256"] != contract_sha for audit in audits.values()):
            raise ValueError("Existing index contract mismatch")
        index = pd.read_csv(args.output / "streetview_index.csv", dtype={"panoid": str, "capture_month": str})
        if len(index) != sum(audit["matched_metadata_panos"] for audit in audits.values()):
            raise ValueError("Existing master index does not match completed city indexes")
    else:
        work = [(city, contract["cities"][city], args.raw_root, args.output, contract_sha, args.metadata_only_index) for city in cities]
        with ProcessPoolExecutor(args.city_workers) as pool:
            results = list(pool.map(process_city_work, work))
        frames = [frame for _, frame, _ in results]
        audits = {city: audit for city, _, audit in results}
        index = pd.concat(frames, ignore_index=True)
        index.to_csv(args.output / "streetview_index.csv", index=False)
    events = events.loc[events.city_key.isin(cities)]
    if args.metadata_only_index:
        pre, quality = select_verified_pre(events, index, contract, args.raw_root, args.max_images, args.workers)
    else:
        pre = select_pre_assets(events, index, args.max_images)
        with ThreadPoolExecutor(args.workers) as pool:
            checked = list(pool.map(verify_image, sorted(pre.asset_path.unique())))
        quality = pd.DataFrame(checked)
        if not quality.empty:
            pre = pre.merge(quality.drop(columns="asset_bytes"), on="asset_path", validate="many_to_one")
            pre = pre.loc[pre.image_valid].copy()
    quality.to_csv(args.output / "selected_image_quality.csv", index=False)
    pre.to_csv(args.output / "streetview_pre_selected.csv", index=False)
    counts = pre.groupby("treatment_order").size()
    coverage = events[["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]].copy()
    coverage["streetview_available"] = coverage.treatment_order.map(counts).fillna(0).gt(0)
    coverage["selected_image_count"] = coverage.treatment_order.map(counts).fillna(0).astype(int)
    coverage.to_csv(args.output / "event_coverage.csv", index=False)
    summary = {"schema": "streetview_fixed_treated_grid_v2", "run_finished": True,
               "created_at_utc": datetime.now(UTC).isoformat(), "raw_root": str(args.raw_root),
               "contract_sha256": sha256(args.contract), "treatment_sha256": contract["treatment_sha256"],
               "coordinate_source": "sdata_ret_bd_xy_scaled_to_mc_then_bd09_gcj02_wgs84",
               "capture_date_precision": "month", "temporal_rule": "capture_month < opening_month",
               "spatial_support": "unique_actual_pano_point_in_frozen_treated_grid_polygon",
               "cities": len(cities), "events": len(events), "index_rows": len(index),
               "index_image_availability": "unverified_metadata_candidates" if args.metadata_only_index else "file_presence_verified",
               "available_pre_events": int(coverage.streetview_available.sum()),
               "selected_rows": len(pre), "selected_unique_images": int(pre.asset_path.nunique()),
               "selected_unique_bytes": int(pre.drop_duplicates("asset_path").asset_bytes.sum()),
               "selected_invalid_images": 0,
               "failed_image_lookup_or_decode_attempts": 0 if quality.empty else int((~quality.image_valid).sum()),
               "all_indexed_images_fully_decoded": False, "selected_images_fully_decoded": True,
               "per_city": audits,
               "file_sha256": {name: sha256(args.output / name) for name in
                   ["streetview_index.csv", "streetview_pre_selected.csv", "event_coverage.csv", "selected_image_quality.csv"]}}
    (args.output / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ["per_city", "file_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
