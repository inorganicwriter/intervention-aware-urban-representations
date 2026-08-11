"""Source discovery and archive helpers for the Amap POI asset."""

from __future__ import annotations

import re
import shutil
import tempfile
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pandas as pd

from urban_intervention.config.project import CITIES
from urban_intervention.data.paths import PROJECT_ROOT

from .config import INVENTORY_PATH, POI_DIR, POI_ZIP
from .taxonomy import map_poi_category


def fix_zip_name(name: str) -> str:
    """Recover Chinese filenames affected by common zip mojibake."""
    out = name
    with suppress(Exception):
        out = out.encode("cp437").decode("gbk")
    with suppress(Exception):
        out = out.encode("gbk").decode("utf-8")
    return out


def classify_poi_source(path: str) -> str:
    p = Path(path)
    s = str(path)
    if s.endswith("_wgs84.csv"):
        return "city_csv"
    if s.endswith(".gdb") or p.suffix == ".gdb":
        return "filegdb_dir"
    if s.endswith(".gdb.zip"):
        return "filegdb_zip"
    if s.endswith(".gdb.rar"):
        return "filegdb_rar"
    return "other"


def category_from_gdb_path(path: str) -> str:
    name = Path(path).name
    name = name.replace("全国", "")
    name = re.sub(r"\d{4}", "", name)
    for suffix in [".gdb.zip", ".gdb.rar", ".gdb"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def iter_asset_members(zip_path: Path = POI_ZIP) -> list[dict]:
    rows = []
    with ZipFile(zip_path) as zf:
        for info in zf.infolist():
            fixed = fix_zip_name(info.filename)
            if info.is_dir() or fixed.startswith("__MACOSX/") or fixed.endswith(".DS_Store"):
                continue
            year = _infer_year(fixed)
            suffix = (
                "".join(Path(fixed).suffixes[-2:])
                if fixed.endswith(".gdb.zip")
                else Path(fixed).suffix
            )
            rows.append(
                {
                    "zip_name": info.filename,
                    "path": fixed,
                    "year": year,
                    "suffix": suffix.lower(),
                    "source_type": classify_poi_source(fixed),
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                }
            )
    return rows


def iter_extracted_members(source_dir: Path = POI_DIR) -> list[dict]:
    rows: list[dict] = []
    if not source_dir.exists():
        return rows
    for path in source_dir.rglob("*"):
        if path.name == ".DS_Store" or path.name.startswith("._"):
            continue
        if any(parent.suffix == ".gdb" for parent in path.parents):
            continue

        rel = path.relative_to(source_dir.parent)
        rel_s = str(rel)
        if path.is_dir():
            if path.suffix == ".gdb":
                rows.append(
                    {
                        "zip_name": "",
                        "path": rel_s,
                        "year": _infer_year(rel_s),
                        "suffix": ".gdb",
                        "source_type": "filegdb_dir",
                        "file_size": _directory_size(path),
                        "compress_size": _directory_size(path),
                        "source": "extracted",
                    }
                )
            continue
        if not path.is_file():
            continue

        if re.search(r"/20(1[2-7])-84\.zip$", rel_s.replace("\\", "/")):
            rows.extend(_iter_nested_city_csv_members(path, rel_s))
            continue

        suffix = "".join(path.suffixes[-2:]) if str(path).endswith(".gdb.zip") else path.suffix
        rows.append(
            {
                "zip_name": "",
                "path": rel_s,
                "year": _infer_year(rel_s),
                "suffix": suffix.lower(),
                "source_type": classify_poi_source(rel_s),
                "file_size": path.stat().st_size,
                "compress_size": path.stat().st_size,
                "source": "extracted",
            }
        )
    return rows


def _directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _iter_nested_city_csv_members(zip_path: Path, rel_s: str) -> list[dict]:
    rows = []
    with ZipFile(zip_path) as zf:
        for info in zf.infolist():
            fixed = fix_zip_name(info.filename)
            if info.is_dir() or not fixed.endswith(".csv"):
                continue
            full_path = f"{rel_s}:{fixed}"
            rows.append(
                {
                    "zip_name": rel_s,
                    "path": full_path,
                    "year": _infer_year(full_path),
                    "suffix": ".csv",
                    "source_type": classify_poi_source(fixed),
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "source": "extracted_zip_member",
                }
            )
    return rows


def write_inventory(zip_path: Path = POI_ZIP, out_path: Path = INVENTORY_PATH) -> pd.DataFrame:
    rows = iter_extracted_members()
    if zip_path.exists():
        zip_rows = iter_asset_members(zip_path)
        for row in zip_rows:
            row["source"] = "zip"
        rows.extend(zip_rows)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved inventory -> {out_path} ({len(df)} files)")
    return df


def _infer_year(path: str) -> int | None:
    s = str(path).replace("\\", "/")
    # The "{year}-84" directory (or zip) segment is the authoritative year
    # marker for Amap city CSV trees.  Plain first-match scanning would grab
    # "2012" from a parent directory like "2012-2017-WGS84" and mislabel every
    # 2013-2017 file as 2012.
    year_dir = re.search(r"(\d{4})-84(?:/|\.zip)", s)
    if year_dir:
        return int(year_dir.group(1))
    scopes = []
    if ":" in s:
        scopes.append(s.rsplit(":", 1)[1])
    scopes.extend([s.rsplit("/", 1)[-1], s])
    for scope in scopes:
        years = [int(match) for match in re.findall(r"20(?:1[2-9]|2[0-4])", scope)]
        if years:
            return years[0]
    return None


def city_name_variants(city_key: str) -> list[str]:
    name = CITIES[city_key]["name"]
    variants = [name]
    if not name.endswith(("市", "盟", "区")):
        variants.append(f"{name}市")
    return variants


def matches_city_csv(path: str, city_key: str, year: int) -> bool:
    if not path.endswith(".csv") or f"{year}-84/" not in path:
        return False
    return any(f"/{v}_wgs84.csv" in path for v in city_name_variants(city_key))


def find_disk_city_csv(city_key: str, year: int) -> Path | None:
    if not POI_DIR.exists():
        return None
    year_dir = POI_DIR / "2012-2017-WGS84" / f"{year}-84"
    for variant in city_name_variants(city_key):
        path = year_dir / f"{variant}_wgs84.csv"
        if path.exists():
            return path
    return None


def find_disk_nested_year_zip(year: int) -> Path | None:
    path = POI_DIR / "2012-2017-WGS84" / f"{year}-84.zip"
    return path if path.exists() else None


@contextmanager
def open_city_csv(city_key: str, year: int):
    disk_csv = find_disk_city_csv(city_key, year)
    if disk_csv is not None:
        with disk_csv.open("rb") as fh:
            try:
                source_name = str(disk_csv.relative_to(PROJECT_ROOT))
            except ValueError:
                source_name = str(disk_csv.resolve())
            yield fh, source_name
        return

    disk_zip = find_disk_nested_year_zip(year)
    if disk_zip is not None:
        with ZipFile(disk_zip) as nested:
            for info in nested.infolist():
                fixed = fix_zip_name(info.filename)
                if matches_city_csv(fixed, city_key, year):
                    with nested.open(info) as fh:
                        yield fh, f"{disk_zip}:{fixed}"
                    return

    if not POI_ZIP.exists():
        raise FileNotFoundError(f"No CSV source found for {city_key} {year}")
    with _open_city_csv_from_outer_zip(city_key, year) as opened:
        yield opened


@contextmanager
def _open_city_csv_from_outer_zip(city_key: str, year: int):
    with ExitStack() as stack:
        outer = stack.enter_context(ZipFile(POI_ZIP))
        direct = _find_direct_city_csv(outer, city_key, year)
        if direct is not None:
            fh = stack.enter_context(outer.open(direct))
            yield fh, fix_zip_name(direct.filename)
            return

        nested_info = _find_nested_year_zip(outer, year)
        if nested_info is None:
            raise FileNotFoundError(f"No nested zip found for year {year}")

        tmp_name = stack.enter_context(tempfile.TemporaryDirectory(prefix=f"poi_{year}_"))
        nested_path = Path(tmp_name) / f"{year}-84.zip"
        with outer.open(nested_info) as src, nested_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024 * 8)
        nested = stack.enter_context(ZipFile(nested_path))
        for info in nested.infolist():
            fixed = fix_zip_name(info.filename)
            if matches_city_csv(fixed, city_key, year):
                fh = stack.enter_context(nested.open(info))
                yield fh, fixed
                return
        raise FileNotFoundError(f"No city CSV found for {city_key} {year}")


def _find_direct_city_csv(zf: ZipFile, city_key: str, year: int) -> ZipInfo | None:
    for info in zf.infolist():
        fixed = fix_zip_name(info.filename)
        if fixed.startswith("__MACOSX/"):
            continue
        if matches_city_csv(fixed, city_key, year):
            return info
    return None


def _find_nested_year_zip(zf: ZipFile, year: int) -> ZipInfo | None:
    target = f"{year}-84.zip"
    for info in zf.infolist():
        fixed = fix_zip_name(info.filename)
        if fixed.startswith("__MACOSX/"):
            continue
        if fixed.endswith(target):
            return info
    return None


def find_gdb_sources(year: int, categories: set[str] | None = None) -> list[Path]:
    if not POI_DIR.exists():
        return []
    if year == 2020:
        candidates = [POI_DIR / "2020全国大类.gdb.rar"]
    else:
        year_dir = POI_DIR / str(year)
        if not year_dir.exists():
            return []
        candidates = []
        for path in year_dir.iterdir():
            s = str(path)
            if (
                path.is_dir()
                and s.endswith(".gdb")
                or path.is_file()
                and (s.endswith(".gdb.zip") or s.endswith(".gdb.rar"))
            ):
                candidates.append(path)
    filtered = [p for p in candidates if p.exists() and _matches_categories(p, categories)]
    best: dict[str, Path] = {}
    rank = {"filegdb_dir": 0, "filegdb_zip": 1, "filegdb_rar": 2, "other": 3}
    for path in filtered:
        cat = category_from_gdb_path(str(path))
        prev = best.get(cat)
        if (
            prev is None
            or rank[classify_poi_source(str(path))] < rank[classify_poi_source(str(prev))]
        ):
            best[cat] = path
    return list(best.values())


def _matches_categories(path: Path, categories: set[str] | None) -> bool:
    if not categories:
        return True
    cat = category_from_gdb_path(str(path))
    return cat in categories or map_poi_category(cat) in categories


@contextmanager
def extract_gdb_zip_to_temp(path: Path):
    tmpdir = tempfile.TemporaryDirectory(prefix="poi_gdb_zip_")
    try:
        root = Path(tmpdir.name)
        with ZipFile(path) as zf:
            names = zf.namelist()
            root_has_gdb_files = any(name.endswith(".gdbtable") for name in names)
            if root_has_gdb_files:
                gdb_dir = root / path.name.replace(".zip", "")
                gdb_dir.mkdir(parents=True, exist_ok=True)
                zf.extractall(gdb_dir)
            else:
                zf.extractall(root)
                gdb_dir = None
        gdbs = [gdb_dir] if gdb_dir is not None else list(root.rglob("*.gdb"))
        if not gdbs:
            raise FileNotFoundError(f"No .gdb directory found inside {path}")
        yield gdbs[0], tmpdir
    finally:
        tmpdir.cleanup()


def choose_archive_tool() -> str | None:
    for name in ["7z", "7za", "unrar"]:
        if shutil.which(name):
            return name
    return None
