"""Package Baidu street view images and metadata by city.

Directly copies province/city subdirectories from raw data into structured
output per target city.  No CSV parsing, no grid matching.

Usage (on server):
    python package_streetview.py --city all
    python package_streetview.py --city beijing,shanghai
    python package_streetview.py --city beijing --clean

Output:
    output/streetview/{city}/
        metadata/   — CSV files
        images/     — image files
"""

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from urban_intervention.data.paths import OUTPUT_STREETVIEW_DIR

ROOT = Path(__file__).resolve().parents[2]

CITIES = {
    "beijing": ("北京", "Beijing"),
    "changchun": ("长春", "Jilin"),
    "changsha": ("长沙", "Hunan"),
    "changzhou": ("常州", "Jiangsu"),
    "chengdu": ("成都", "Sichuan"),
    "chongqing": ("重庆", "Chongqing"),
    "dalian": ("大连", "Liaoning"),
    "dongguan": ("东莞", "Guangdong"),
    "foshan": ("佛山", "Guangdong"),
    "fuzhou": ("福州", "Fujian"),
    "guangzhou": ("广州", "Guangdong"),
    "guiyang": ("贵阳", "Guizhou"),
    "hangzhou": ("杭州", "Zhejiang"),
    "harbin": ("哈尔滨", "Heilongjiang"),
    "hefei": ("合肥", "Anhui"),
    "hohhot": ("呼和浩特", "Neimenggu"),
    "jinan": ("济南", "Shandong"),
    "jinhua": ("金华", "Zhejiang"),
    "kunming": ("昆明", "Yunnan"),
    "lanzhou": ("兰州", "Gansu"),
    "luoyang": ("洛阳", "Henan"),
    "nanchang": ("南昌", "Jiangxi"),
    "nanjing": ("南京", "Jiangsu"),
    "nanning": ("南宁", "Guangxi"),
    "nantong": ("南通", "Jiangsu"),
    "ningbo": ("宁波", "Zhejiang"),
    "qingdao": ("青岛", "Shandong"),
    "shanghai": ("上海", "Shanghai"),
    "shaoxing": ("绍兴", "Zhejiang"),
    "shenyang": ("沈阳", "Liaoning"),
    "shenzhen": ("深圳", "Guangdong"),
    "shijiazhuang": ("石家庄", "Hebei"),
    "suzhou": ("苏州", "Jiangsu"),
    "taiyuan": ("太原", "Shanxi"),
    "taizhou": ("台州", "Zhejiang"),
    "tianjin": ("天津", "Tianjin"),
    "urumqi": ("乌鲁木齐", "Xinjiang"),
    "wenzhou": ("温州", "Zhejiang"),
    "wuhan": ("武汉", "Hubei"),
    "wuxi": ("无锡", "Jiangsu"),
    "xiamen": ("厦门", "Fujian"),
    "xian": ("西安", "Shaanxi"),
    "xuzhou": ("徐州", "Jiangsu"),
    "zhengzhou": ("郑州", "Henan"),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def find_city_dirs(
    base_dir: str,
    province: str,
    city_key: str,
    file_suffixes: set | None = None,
    data_label: str = "data",
) -> list[Path]:
    """Find directories containing data for a city under a province.

    Two directory structures are supported:
    A) {province}/{city}/  — city has its own subdirectory
    B) {province}/         — data files directly in the province dir

    For structure B, only returns the province dir if the province
    contains at most ONE of our target cities (safe for single-city
    provinces and municipalities).
    """
    base = Path(base_dir) / province
    if not base.is_dir():
        return []

    cn_name = CITIES[city_key][0]
    candidate = {city_key.casefold(), cn_name.casefold()}

    # Strategy A: find subdirectory matching our city name
    matches = [
        path for path in base.iterdir() if path.is_dir() and path.name.casefold() in candidate
    ]

    if matches:
        return matches

    # Strategy B: no city subdirectory — data is directly in the province dir
    files = list(base.iterdir())
    if not any(f.is_file() for f in files):
        return []

    if file_suffixes is not None:
        has_relevant = any(f.is_file() and _suffix_match(f.suffix, file_suffixes) for f in files)
        if not has_relevant:
            return []

    # Count how many of our target cities are in this province
    province_cities = [ck for ck, (_, prov) in CITIES.items() if prov == province]

    if len(province_cities) > 1:
        print(
            f"\n  WARNING: Cannot assign {data_label} in {base} to "
            f"{city_key}: {province} has {len(province_cities)} target "
            f"cities but no city subdirectory.",
            flush=True,
        )
        return []

    return [base]


def _suffix_match(suffix: str, allowed: set) -> bool:
    """Case-insensitive suffix check."""
    return suffix.casefold() in {s.casefold() for s in allowed}


_RSYNC_AVAILABLE: bool | None = None  # cached


def _has_rsync() -> bool:
    global _RSYNC_AVAILABLE
    if _RSYNC_AVAILABLE is None:
        import subprocess

        try:
            subprocess.run(["rsync", "--version"], capture_output=True, check=True)
            _RSYNC_AVAILABLE = True
        except Exception:
            _RSYNC_AVAILABLE = False
    return _RSYNC_AVAILABLE


def _copy_source_dir(src: Path, dst: Path, allowed_suffixes: set | None = None) -> int:
    """Copy files from src into dst.  Uses rsync if available, else Python."""
    if not src.is_dir():
        return 0

    if _has_rsync():
        import subprocess

        dst.mkdir(parents=True, exist_ok=True)
        include = []
        if allowed_suffixes:
            for s in allowed_suffixes:
                include.extend(["--include", f"*{s}", "--include", f"*{s.upper()}"])
            include.extend(["--include", "*/", "--exclude", "*"])
        subprocess.run(
            ["rsync", "-rt", "--info=progress2", *include, f"{src}/", f"{dst}/"],
            check=True,
        )
        return 1

    normalized = {s.casefold() for s in allowed_suffixes} if allowed_suffixes is not None else None
    count = 0
    for item in sorted(src.iterdir()):
        dest = dst / item.name
        if item.is_dir():
            count += _copy_source_dir(item, dest, allowed_suffixes)
        elif item.is_file():
            if normalized is not None and item.suffix.casefold() not in normalized:
                continue
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, dest)
            count += 1
    return count


def package_city(
    city_key: str, metadata_root: str, images_root: str, out_root: str, clean: bool = False
) -> tuple[Path, bool, bool]:
    """Copy metadata and images for one city."""
    cn_name, province = CITIES[city_key]
    city_out = Path(out_root) / city_key

    if clean and city_out.exists():
        shutil.rmtree(city_out)

    meta_dirs = find_city_dirs(
        metadata_root, province, city_key, file_suffixes={".csv"}, data_label="metadata"
    )
    img_dirs = find_city_dirs(
        images_root, province, city_key, file_suffixes=IMAGE_SUFFIXES, data_label="images"
    )

    dest_meta = city_out / "metadata"
    dest_img = city_out / "images"

    sum(_copy_source_dir(d, dest_meta, {".csv"}) for d in meta_dirs)
    sum(_copy_source_dir(d, dest_img, IMAGE_SUFFIXES) for d in img_dirs)

    meta_total = sum(1 for _ in dest_meta.rglob("*") if _.is_file()) if dest_meta.exists() else 0
    img_total = sum(1 for _ in dest_img.rglob("*") if _.is_file()) if dest_img.exists() else 0

    print(
        f"\n  {city_key} ({cn_name}, {province}): meta={meta_total} files, img={img_total} files",
        flush=True,
    )

    return city_out, meta_total > 0, img_total > 0


def main():
    parser = argparse.ArgumentParser(description="Package Baidu street view data by city")
    parser.add_argument(
        "--metadata-dir",
        default=os.environ.get("MIT_STREETVIEW_METADATA_DIR"),
        help="Root of metadata CSV directory",
    )
    parser.add_argument(
        "--images-dir",
        default=os.environ.get("MIT_STREETVIEW_IMAGES_DIR"),
        help="Root of street view images directory",
    )
    parser.add_argument("--out-dir", default=str(OUTPUT_STREETVIEW_DIR), help="Output directory")
    parser.add_argument("--city", default="all", help="City key, comma-separated, or 'all'")
    parser.add_argument(
        "--clean", action="store_true", help="Delete existing city output before copying"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of cities to process in parallel (default: 1)",
    )
    args = parser.parse_args()

    if not args.metadata_dir:
        parser.error("Provide --metadata-dir or set MIT_STREETVIEW_METADATA_DIR")
    if not args.images_dir:
        parser.error("Provide --images-dir or set MIT_STREETVIEW_IMAGES_DIR")

    # Validate city names
    if args.city == "all":
        target = list(CITIES.keys())
    else:
        requested = [c.strip().casefold() for c in args.city.split(",") if c.strip()]
        # Map back to canonical case (CITIES keys are lowercase)
        target = []
        invalid = []
        for name in requested:
            if name in CITIES:
                target.append(name)
            else:
                invalid.append(name)
        if invalid:
            parser.error(
                "Unknown city key(s): "
                + ", ".join(invalid)
                + "\nValid keys: "
                + ", ".join(sorted(CITIES.keys()))
            )

    if not target:
        parser.error("No valid cities specified")

    print(f"\nTargets: {len(target)} cities")
    print(f"Metadata: {args.metadata_dir}")
    print(f"Images:   {args.images_dir}")
    print(f"Output:   {args.out_dir}")
    if args.clean:
        print("Clean:    yes")
    print(f"Workers:  {args.workers}")

    results = {}
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    package_city,
                    city_key,
                    args.metadata_dir,
                    args.images_dir,
                    args.out_dir,
                    args.clean,
                ): city_key
                for city_key in target
            }
            for future in as_completed(futures):
                city_key = futures[future]
                try:
                    out, has_meta, has_img = future.result()
                    results[city_key] = (out, has_meta, has_img)
                except Exception as e:
                    print(f"\n  ERROR {city_key}: {e}", flush=True)
                    results[city_key] = (Path(args.out_dir) / city_key, False, False)
    else:
        for city_key in target:
            out, has_meta, has_img = package_city(
                city_key, args.metadata_dir, args.images_dir, args.out_dir, clean=args.clean
            )
            results[city_key] = (out, has_meta, has_img)

    # Summary: only show requested cities
    print("\n=== Summary ===")
    for city_key in target:
        city_dir, has_meta, has_img = results[city_key]
        if has_meta and has_img:
            status = "OK"
        elif has_meta:
            status = "METADATA ONLY"
        elif has_img:
            status = "IMAGES ONLY"
        else:
            status = "EMPTY"
        print(f"  {city_key}: {status}")

    print(f"\nDone. Output: {args.out_dir}")
    print(f"To download: tar -czf streetview.tar.gz {args.out_dir}")


if __name__ == "__main__":
    main()
