"""Cross-source comparison & validation for metro station data.

For each of the 44 cities, loads every available source CSV in
data/archive/raw/transit/{city}/, computes coverage & overlap metrics, picks the
most comprehensive source, and writes a unioned "merged" table with
provenance for downstream use in build_treatment.py.

Source tags are inferred from filename suffix:
  {city}_metro_stations_amap.csv     -> amap
  {city}_metro_stations_osm.csv      -> osm
  {city}_metro_stations_wikidata.csv -> wikidata
  {city}_stations_wiki.csv           -> wiki
  {city}_metro_stations_github*.csv  -> github_*

Outputs (under outputs/transit_comparison/):
  per_city_summary.csv     — city × source: count, n_year, n_line, n_transfer, score
  per_city_overlap.csv     — pairwise name-overlap counts per city
  best_source_per_city.csv — recommended source + rationale per city
  per_city_anomalies.csv   — stations found in only one source, or name-matched
                              across sources with coords >500m apart

Also writes per city:
  data/archive/raw/transit/{city}/{city}_metro_stations_merged.csv
  (union of all sources, deduped by normalized name + 100m coord cluster,
   with a 'sources' column listing which sources contributed and a
   'opening_year' taken as the min non-null across sources.)

Usage:
    python scripts/collection/compare_transit_sources.py
    python scripts/collection/compare_transit_sources.py --city beijing
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import numpy as np
import pandas as pd

from urban_intervention.config.project import ACTIVE_CITIES
from urban_intervention.config.project import norm_station_name as _norm
from urban_intervention.data.paths import RAW_TRANSIT_DIR

RAW = RAW_TRANSIT_DIR
OUT = BASE_DIR / "outputs" / "transit_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# Approximate official operating-station counts (end of 2024) for sanity.
# Marked as approximate; used only as a soft benchmark, never as ground truth.
# Sources: official municipal / operator announcements + Wikipedia
# "List of metro systems" + China MoT 2025-05 statistics, cross-checked.
# A value of 0 means "no reliable reference" — the score function treats
# falsy values the same as None (falls back to relative coverage).
OFFICIAL_REF = {
    # Tier 1 — large systems (>150 stations)
    "beijing": 490,
    "shanghai": 508,
    "guangzhou": 402,
    "shenzhen": 411,
    "chengdu": 413,
    "wuhan": 411,
    "hangzhou": 260,
    "nanjing": 217,
    "chongqing": 256,
    "xian": 193,
    "suzhou": 247,
    "zhengzhou": 224,
    "tianjin": 195,
    "qingdao": 172,
    "changsha": 219,
    "kunming": 165,
    # Tier 2 — medium systems (50-150 stations)
    "dalian": 68,
    "ningbo": 102,
    "hefei": 109,
    "nanning": 87,
    "fuzhou": 80,
    "xiamen": 98,
    "wuxi": 75,
    "guiyang": 102,
    "shijiazhuang": 76,
    "changchun": 96,
    "jinan": 91,
    "harbin": 78,
    "shenyang": 102,  # 10 lines, ~102 stations end of 2024
    "foshan": 47,
    "dongguan": 47,
    "xuzhou": 64,
    "nanchang": 85,
    "wenzhou": 46,
    "luoyang": 30,
    "shaoxing": 32,
    "nantong": 28,
    "lanzhou": 26,
    "taiyuan": 36,
    "urumqi": 21,
    "hohhot": 43,
    "taizhou": 30,
    "jinhua": 0,
    "changzhou": 29,  # Changzhou metro line 1 opened 2020-09
}

SOURCE_TAGS = ["amap", "osm", "wikidata", "wiki"]  # github_* matched dynamically


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    R = 6371000
    p1, p2 = np.radians([lat1, lon1]), np.radians([lat2, lon2])
    dlat, dlon = p2[0] - p1[0], p2[1] - p1[1]
    a = np.sin(dlat / 2) ** 2 + np.cos(p1[0]) * np.cos(p2[0]) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def _tag_from_filename(fname: str, city_key: str) -> str | None:
    """Infer source tag from filename. Returns None for unrelated files."""
    stem = Path(fname).stem
    prefix = f"{city_key}_metro_stations_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    if stem == f"{city_key}_stations_wiki":
        return "wiki"
    return None


def _load_source(city_key: str) -> dict[str, pd.DataFrame]:
    """Return {tag: df} for every source file present for this city."""
    sources = {}
    if not RAW.exists():
        return sources
    for f in sorted(RAW.glob(f"*/{city_key}/*.csv")):
        tag = _tag_from_filename(f.name, city_key)
        if tag is None:
            continue
        # skip the merged output we ourselves write
        if tag == "merged":
            continue
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception as e:
            print(f"    [skip] {f.name}: {e}")
            continue
        if df.empty or "station_name" not in df.columns:
            continue
        df["_n"] = df["station_name"].apply(_norm)
        df["_clat"] = df["wgs84_lat"].round(3) if "wgs84_lat" in df else np.nan
        df["_clon"] = df["wgs84_lon"].round(3) if "wgs84_lon" in df else np.nan
        sources[tag] = df
    return sources


def _stats(df: pd.DataFrame) -> dict:
    n = len(df)
    n_year = int(df["opening_year"].notna().sum()) if "opening_year" in df else 0
    n_line = (
        int((df.get("line", pd.Series(dtype=str)).fillna("").astype(str) != "").sum())
        if "line" in df
        else 0
    )
    n_transfer = 0
    if "line" in df:
        n_transfer = int(
            df["line"]
            .fillna("")
            .astype(str)
            .str.split(";")
            .apply(lambda xs: len([x for x in xs if x]) >= 2)
            .sum()
        )
    return {"count": n, "n_year": n_year, "n_line": n_line, "n_transfer": n_transfer}


def _pairwise_overlap(sources: dict[str, pd.DataFrame]) -> list[dict]:
    """For each pair of sources, count normalized-name matches (intersection)."""
    tags = list(sources.keys())
    rows = []
    for i, a in enumerate(tags):
        sa = set(sources[a]["_n"].dropna())
        for b in tags[i + 1 :]:
            sb = set(sources[b]["_n"].dropna())
            inter = len(sa & sb)
            rows.append(
                {
                    "src_a": a,
                    "src_b": b,
                    "overlap": inter,
                    "only_a": len(sa - sb),
                    "only_b": len(sb - sa),
                    "jaccard": round(inter / max(1, len(sa | sb)), 3),
                }
            )
    return rows


def _coord_agreement(sources: dict[str, pd.DataFrame]) -> list[dict]:
    """Flag name-matched stations whose coords disagree across sources (>500m)."""
    # Build long table: station_name_norm, source, lon, lat
    long = []
    for tag, df in sources.items():
        if "wgs84_lon" not in df:
            continue
        for _, r in df.iterrows():
            long.append({"_n": r["_n"], "src": tag, "lon": r["wgs84_lon"], "lat": r["wgs84_lat"]})
    if len(long) < 2:
        return []
    ldf = pd.DataFrame(long)
    anomalies = []
    for n, grp in ldf.groupby("_n"):
        if len(grp) < 2:
            continue
        coords = grp[["lon", "lat"]].values
        maxd = 0.0
        worst = None
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = _haversine_m(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
                if d > maxd:
                    maxd = d
                    worst = (grp.iloc[i]["src"], grp.iloc[j]["src"])
        if maxd > 500:
            anomalies.append(
                {"station": n, "max_dist_m": round(maxd, 1), "src_pair": f"{worst[0]}|{worst[1]}"}
            )
    return anomalies


def _singletons(sources: dict[str, pd.DataFrame]) -> list[dict]:
    """Stations found in exactly one source (potential typos / phantom POIs)."""
    from collections import defaultdict

    seen: dict[str, list[str]] = defaultdict(list)
    for tag, df in sources.items():
        for n in df["_n"].dropna().unique():
            seen[n].append(tag)
    return [{"station": n, "src": ts[0]} for n, ts in seen.items() if len(ts) == 1]


def _score(stats: dict, max_count: int, official_ref: int | None) -> float:
    """Completeness score in [0,1].

    When an official reference count is known, count accuracy is weighted most
    heavily — for a treatment matrix, including wrong-city stations (over-count)
    is worse than missing opening years, because it corrupts the treatment
    definition itself. Coverage is capped at the reference so over-counting
    never helps. When no reference, fall back to coverage vs other sources.
    """
    count = stats["count"]
    if count == 0:
        return 0.0
    year_fill = stats["n_year"] / count
    line_fill = stats["n_line"] / count
    if official_ref and official_ref > 0:
        accuracy = max(0.0, 1.0 - abs(count - official_ref) / official_ref)
        coverage = min(count, official_ref) / official_ref  # capped at ref
        return round(0.60 * accuracy + 0.20 * coverage + 0.10 * year_fill + 0.10 * line_fill, 3)
    # No reference: relative coverage vs best available source
    coverage = count / max_count if max_count else 0.0
    return round(0.50 * coverage + 0.30 * year_fill + 0.20 * line_fill, 3)


def _reliable(count: int, official_ref: int | None) -> str:
    """Flag whether a source's count is within 15% of the official reference."""
    if not official_ref or official_ref == 0:
        return "unknown"
    delta = abs(count - official_ref) / official_ref
    return "yes" if delta < 0.15 else "no"


def _merge_sources(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Union all sources, dedup by (_n, _clat, _clon); track provenance.

    When multiple sources provide an opening date for the same station, the
    finest available precision is kept (day > month > year).  This preserves
    the month/day columns introduced by the upgraded Wikidata and Wikipedia
    fetchers instead of collapsing to year-only.
    """
    frames = []
    for tag, df in sources.items():
        sub = df.copy()
        sub["source"] = tag
        # Ensure month/day/precision columns exist even for legacy sources.
        for col in ("opening_month", "opening_day", "opening_date", "date_precision"):
            if col not in sub.columns:
                sub[col] = pd.NA
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    big = pd.concat(frames, ignore_index=True, sort=False)
    keep = [
        "station_name",
        "name_en",
        "wgs84_lon",
        "wgs84_lat",
        "opening_year",
        "opening_month",
        "opening_day",
        "opening_date",
        "date_precision",
        "line",
        "source",
        "_n",
        "_clat",
        "_clon",
    ]
    big = big[[c for c in keep if c in big.columns]]

    # Coerce numeric date columns so min/first aggregation works cleanly.
    for col in ("opening_year", "opening_month", "opening_day"):
        big[col] = pd.to_numeric(big[col], errors="coerce")

    def _pick_finest_date(group: pd.DataFrame) -> pd.Series:
        """Pick the finest-precision date row for a collapsed station."""
        # Rank precision: day (3) > month (2) > year (1) > none (0).
        ranks = group["date_precision"].map({"day": 3, "month": 2, "year": 1}).fillna(0)
        best_idx = ranks.idxmax()
        best = group.loc[best_idx]
        return pd.Series(
            {
                "opening_year": best["opening_year"],
                "opening_month": best["opening_month"],
                "opening_day": best["opening_day"],
                "opening_date": best["opening_date"] if pd.notna(best["opening_date"]) else "",
                "date_precision": best["date_precision"]
                if pd.notna(best["date_precision"])
                else "",
            }
        )

    # Aggregate per (_n, _clat, _clon): first coords/name, min year, union lines, union sources.
    base = big.groupby(["_n", "_clat", "_clon"], as_index=False).agg(
        {
            "station_name": "first",
            "name_en": "first",
            "wgs84_lon": "first",
            "wgs84_lat": "first",
            "line": lambda s: ";".join(sorted({x for y in s for x in str(y).split(";") if x})),
            "source": lambda s: ";".join(sorted(set(s))),
        }
    )

    # Compute finest date per group separately (agg can't easily do conditional logic).
    date_cols = big.groupby(["_n", "_clat", "_clon"]).apply(_pick_finest_date).reset_index()
    date_cols = date_cols.drop(columns=["level_2"], errors="ignore")

    merged = base.merge(date_cols, on=["_n", "_clat", "_clon"], how="left")
    merged = merged.drop(columns=["_n", "_clat", "_clon"]).reset_index(drop=True)
    merged["n_sources"] = merged["source"].str.split(";").apply(len)
    return merged


def run_city(ck: str, summary_rows, overlap_rows, anomaly_rows, singleton_rows, best_rows) -> None:
    sources = _load_source(ck)
    if not sources:
        print(f"  [{ck}] no source files")
        return
    print(f"\n[{ck}] {len(sources)} sources: {', '.join(sorted(sources))}")

    stats = {tag: _stats(df) for tag, df in sources.items()}
    max_count = max(s["count"] for s in stats.values()) if stats else 0
    ref = OFFICIAL_REF.get(ck)

    for tag, st in stats.items():
        score = _score(st, max_count, ref)
        summary_rows.append(
            {
                "city": ck,
                "source": tag,
                "count": st["count"],
                "n_year": st["n_year"],
                "n_line": st["n_line"],
                "n_transfer": st["n_transfer"],
                "year_pct": round(st["n_year"] / max(1, st["count"]), 3),
                "line_pct": round(st["n_line"] / max(1, st["count"]), 3),
                "score": score,
                "official_ref": ref if ref else "",
                "delta_vs_ref": (st["count"] - ref) if ref else "",
            }
        )

    # Best source = highest score (tie-break: closest to official ref, then count)
    ranking = sorted(
        stats.items(),
        key=lambda kv: (
            _score(kv[1], max_count, ref),
            -(abs(kv[1]["count"] - ref) if ref else 0),
            kv[1]["count"],
            kv[1]["n_year"],
        ),
        reverse=True,
    )
    best_tag, best_st = ranking[0]
    best_rows.append(
        {
            "city": ck,
            "best_source": best_tag,
            "best_count": best_st["count"],
            "best_year_pct": round(best_st["n_year"] / max(1, best_st["count"]), 3),
            "official_ref": ref if ref else "",
            "delta_vs_ref": (best_st["count"] - ref) if ref else "",
            "best_reliable": _reliable(best_st["count"], ref),
            "score": _score(best_st, max_count, ref),
            "n_sources_available": len(sources),
            "all_sources": ";".join(sorted(sources)),
        }
    )

    for ov in _pairwise_overlap(sources):
        ov["city"] = ck
        overlap_rows.append(ov)
    for an in _coord_agreement(sources):
        an["city"] = ck
        anomaly_rows.append(an)
    for sg in _singletons(sources):
        sg["city"] = ck
        singleton_rows.append(sg)

    # Write merged union
    merged = _merge_sources(sources)
    if not merged.empty:
        out = RAW / "merged" / ck / f"{ck}_metro_stations_merged.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  merged -> {out} ({len(merged)} unique, {merged['n_sources'].max()} sources max)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="all")
    args = p.parse_args()
    cities = ACTIVE_CITIES if args.city == "all" else [args.city]

    summary_rows, overlap_rows, anomaly_rows, singleton_rows, best_rows = ([], [], [], [], [])
    for ck in cities:
        run_city(ck, summary_rows, overlap_rows, anomaly_rows, singleton_rows, best_rows)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            OUT / "per_city_summary.csv", index=False, encoding="utf-8-sig"
        )
    if overlap_rows:
        pd.DataFrame(overlap_rows).to_csv(
            OUT / "per_city_overlap.csv", index=False, encoding="utf-8-sig"
        )
    if anomaly_rows:
        pd.DataFrame(anomaly_rows).to_csv(
            OUT / "per_city_anomalies.csv", index=False, encoding="utf-8-sig"
        )
    if singleton_rows:
        pd.DataFrame(singleton_rows).to_csv(
            OUT / "per_city_singletons.csv", index=False, encoding="utf-8-sig"
        )
    if best_rows:
        pd.DataFrame(best_rows).to_csv(
            OUT / "best_source_per_city.csv", index=False, encoding="utf-8-sig"
        )
        print("\n=== Best source per city ===")
        br = pd.DataFrame(best_rows)
        for _, r in br.iterrows():
            ref = (
                f" (ref {r['official_ref']}, Δ {r['delta_vs_ref']}, {r['best_reliable']})"
                if r["official_ref"] != ""
                else ""
            )
            print(
                f"  {r['city']:14s} -> {r['best_source']:10s} "
                f"count={r['best_count']} score={r['score']}{ref}"
            )

    print(f"\nReports written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
