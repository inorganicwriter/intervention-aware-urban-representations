"""Build time-of-opening station/line attributes for treated and all canonical stations.

Derives, for every canonical station event at its opening date:

- ``n_lines_at_opening`` / ``is_transfer_at_opening``: number of lines whose
  first-ever station opened on or before this station's opening date
  (``lines`` in the canonical table is CURRENT network membership, so line
  opening dates are reconstructed from the same table and membership is
  snapshotted at the station's opening date).
- ``is_new_line_opening``: at least one line first opened on this station's
  opening date (the station is part of a brand-new line's initial batch).
- ``is_extension_opening``: at least one line was already operating before
  this station's opening date (extension / infill on an existing line).
- ``is_terminal_at_opening``: the station was an endpoint at its opening.
  Endpoint status uses the station-level adjacency graph from
  ``wikidata_adjacency`` (its ``line_label`` marks the station's line, not the
  edge's line, so degrees are computed across all adjacent stations, which
  equals line degree for non-transfer stations); a station is an endpoint
  when exactly one adjacent station had opened on or before its opening
  date.  Transfer stations (ambiguous across lines) and stations without
  matched adjacency rows get NA.
- ``same_month_openings``: number of station entities (name-deduped) in the
  same city opening in the same calendar month, including the station itself.
- ``same_month_new_line_stations``: subset of the same-month batch that are
  themselves part of a brand-new line opening.

Output: ``outputs/causal_labels/station_attributes/station_attributes.parquet``
(one row per canonical station event, with ``is_treated``/``treatment_order``
flags joined from the frozen 5,048 treatment list) and a diagnostics JSON in
the same directory.

Usage:
    python scripts/analysis/build_station_attributes.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.config.project import CITIES  # noqa: E402

REFERENCE_DIR = ROOT / "data" / "active" / "reference" / "transit"
CAUSAL_DIR = ROOT / "data" / "active" / "causal"
OUTPUT_DIR = ROOT / "outputs" / "causal_labels" / "station_attributes"

STATION_EVENTS_PATH = REFERENCE_DIR / "canonical_station_events_resolved.parquet"
ADJACENCY_PATH = REFERENCE_DIR / "wikidata_adjacency.parquet"
TREATMENTS_PATH = CAUSAL_DIR / "treatment_unit_list.parquet"
OUT_PATH = OUTPUT_DIR / "station_attributes.parquet"
DIAGNOSTICS_PATH = OUTPUT_DIR / "diagnostics.json"

_LINE_SEPARATORS = re.compile(r"[;；]")
_LINE_VARIANTS = re.compile(r"[／/]")
_PUNCT = re.compile(r"[\s·•\-—–_（）()\[\]【】]+")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def norm_station(name: object) -> str:
    text = _nfkc(name)
    text = re.sub(r"\(地铁站\)$", "", text)
    text = re.sub(r"地铁站$", "", text)
    text = re.sub(r"站$", "", text)
    return _PUNCT.sub("", text)


def norm_line(label: object) -> str:
    text = _nfkc(label)
    for city in CITIES.values():
        city_name = str(city.get("name", ""))
        if city_name and text.startswith(city_name):
            text = text[len(city_name):]
            break
    text = re.sub(r"^(地铁|轨道交通|铁)", "", text)
    return text


def split_line_entities(lines_raw: object) -> list[str]:
    if not isinstance(lines_raw, str) or not lines_raw.strip():
        return []
    return [part.strip() for part in _LINE_SEPARATORS.split(lines_raw) if part.strip()]


def line_variants(entity: str) -> list[str]:
    return [v.strip() for v in _LINE_VARIANTS.split(entity) if v.strip()]


def entity_matches_label(entity: str, label: str) -> bool:
    label = norm_line(label)
    if not label:
        return False
    for variant in line_variants(entity):
        variant = norm_line(variant)
        if not variant:
            continue
        if variant == label:
            return True
        if len(variant) >= len(label):
            if variant.endswith(label):
                return True
        else:
            if label.endswith(variant):
                return True
    return False


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(STATION_EVENTS_PATH)
    required = {
        "city_key",
        "station_event_id",
        "canonical_station_name",
        "lines",
        "opening_date",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Canonical station events lack columns: {sorted(missing)}")
    if events["station_event_id"].duplicated().any():
        raise ValueError("Canonical station events have duplicate station_event_id")
    events["opening_date"] = pd.to_datetime(events["opening_date"], errors="coerce")
    if events["opening_date"].isna().any():
        raise ValueError("Canonical station events contain invalid opening dates")

    adjacency = pd.read_parquet(ADJACENCY_PATH)
    required_adj = {
        "station_name",
        "adj_station_name",
        "line_label",
        "city_key",
    }
    missing_adj = required_adj - set(adjacency.columns)
    if missing_adj:
        raise ValueError(f"Adjacency lacks columns: {sorted(missing_adj)}")

    treatments = pd.read_parquet(TREATMENTS_PATH, columns=["treatment_order", "station_event_id"])
    if len(treatments) != 5_048:
        raise ValueError(f"Treatment list is not the immutable 5,048-unit list: {len(treatments)}")
    return events, adjacency, treatments


def build_line_opening_dates(events: pd.DataFrame) -> dict[tuple[str, str], pd.Timestamp]:
    """First-opening date per (city, line variant).

    ``lines`` in the canonical table is CURRENT network membership while a
    station's opening date is its FIRST opening, so a station that joined a
    line later (e.g. an old station now served by a modern line) would make a
    plain min() attribute the line to its own old opening date.  The line's
    opening is instead dated by its first opening BATCH: the earliest date on
    which at least two member stations opened (falling back to the earliest
    member date when every member opened on a distinct day).
    """
    member_dates: dict[tuple[str, str], list[pd.Timestamp]] = {}
    for _, row in events.iterrows():
        city = str(row["city_key"])
        opened = pd.Timestamp(row["opening_date"])
        for entity in split_line_entities(row.get("lines")):
            for variant in line_variants(entity):
                member_dates.setdefault((city, norm_line(variant)), []).append(opened)
    line_first: dict[tuple[str, str], pd.Timestamp] = {}
    for key, dates in member_dates.items():
        counts: dict[pd.Timestamp, int] = {}
        for date in dates:
            counts[date] = counts.get(date, 0) + 1
        ordered = sorted(counts)
        batch = next((date for date in ordered if counts[date] >= 2), None)
        line_first[key] = batch if batch is not None else ordered[0]
    return line_first


def variant_first_dates(
    line_first: dict[tuple[str, str], pd.Timestamp], city: str, entity: str
) -> list[pd.Timestamp]:
    return [
        first
        for variant in line_variants(entity)
        if (first := line_first.get((city, norm_line(variant)))) is not None
    ]


def entity_operating(
    line_first: dict[tuple[str, str], pd.Timestamp],
    city: str,
    entity: str,
    opened: pd.Timestamp,
) -> bool:
    firsts = variant_first_dates(line_first, city, entity)
    return bool(firsts) and min(firsts) <= opened


def compute_station_attributes(
    events: pd.DataFrame, adjacency: pd.DataFrame, treatments: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    line_first = build_line_opening_dates(events)

    city_month_counts: dict[tuple[str, str], int] = {}
    city_month_events: dict[tuple[str, str], list[str]] = {}
    for _, row in events.iterrows():
        key = (str(row["city_key"]), pd.Timestamp(row["opening_date"]).strftime("%Y-%m"))
        city_month_counts[key] = city_month_counts.get(key, 0) + 1
        city_month_events.setdefault(key, []).append(str(row["station_event_id"]))

    name_events: dict[tuple[str, str], list[str]] = {}
    for _, row in events.iterrows():
        norm = norm_station(row["canonical_station_name"])
        if not norm:
            continue
        key = (str(row["city_key"]), norm)
        name_events.setdefault(key, []).append(str(row["station_event_id"]))

    event_to_name: dict[str, str] = {}
    for _, row in events.iterrows():
        event_to_name[str(row["station_event_id"])] = norm_station(
            row["canonical_station_name"]
        )

    event_to_open: dict[str, pd.Timestamp] = {
        str(row["station_event_id"]): pd.Timestamp(row["opening_date"])
        for _, row in events.iterrows()
    }
    event_entities: dict[str, list[str]] = {
        str(row["station_event_id"]): split_line_entities(row.get("lines"))
        for _, row in events.iterrows()
    }

    event_to_treatment: dict[str, int] = {}
    for _, row in treatments.iterrows():
        event_to_treatment[str(row["station_event_id"])] = int(row["treatment_order"])
    found_treatments = len(event_to_treatment)
    if found_treatments != 5_048:
        raise ValueError(
            f"Only {found_treatments}/5,048 treatment events found in canonical events"
        )

    adjacency_by_station: dict[tuple[str, str], set[str]] = {}
    adjacency_matched_count = 0
    for _, row in adjacency.iterrows():
        city = str(row["city_key"])
        s_name = norm_station(row["station_name"])
        a_name = norm_station(row["adj_station_name"])
        if not s_name or not a_name:
            continue
        s_events = name_events.get((city, s_name), [])
        a_events = name_events.get((city, a_name), [])
        if not s_events or not a_events:
            continue
        adjacency_matched_count += 1
        adjacency_by_station.setdefault((city, s_name), set()).add(a_name)
        adjacency_by_station.setdefault((city, a_name), set()).add(s_name)

    records: list[dict[str, object]] = []
    for _, row in events.iterrows():
        eid = str(row["station_event_id"])
        city = str(row["city_key"])
        opened = pd.Timestamp(row["opening_date"])
        entities = event_entities.get(eid, [])
        operating: list[str] = []
        for entity in entities:
            if entity_operating(line_first, city, entity, opened):
                operating.append(entity)
        has_new = any(
            opened in variant_first_dates(line_first, city, entity)
            for entity in entities
            if variant_first_dates(line_first, city, entity)
        )
        has_existing = any(
            min(variant_first_dates(line_first, city, entity)) < opened
            for entity in entities
            if variant_first_dates(line_first, city, entity)
        )

        month_key = (city, opened.strftime("%Y-%m"))
        same_month_ids = city_month_events.get(month_key, [])
        same_month_entities = {
            event_to_name[eid_other] for eid_other in same_month_ids if event_to_name.get(eid_other)
        }
        same_month_new = sum(
            1
            for eid_other in same_month_ids
            if any(
                event_to_open.get(eid_other, opened)
                in variant_first_dates(line_first, city, entity)
                for entity in event_entities.get(eid_other, [])
                if variant_first_dates(line_first, city, entity)
            )
        )

        station_norm = norm_station(row["canonical_station_name"])
        neighbors = adjacency_by_station.get((city, station_norm), set())
        deg_current = 0
        deg_open = 0
        for adj_name in neighbors:
            adj_events = name_events.get((city, adj_name), [])
            if not adj_events:
                continue
            deg_current += 1
            if min(event_to_open[eid] for eid in adj_events) <= opened:
                deg_open += 1
        terminal_value: float | None = None
        terminal_evidence = "no_adjacency"
        if neighbors and deg_current > 0:
            if len(operating) >= 2:
                terminal_evidence = "transfer_degree_ambiguous"
            else:
                terminal_evidence = "station_degree"
                terminal_value = 1.0 if deg_open == 1 else 0.0

        records.append(
            {
                "city_key": city,
                "station_event_id": eid,
                "canonical_station_name": row["canonical_station_name"],
                "opening_date": opened,
                "lines_raw": row.get("lines"),
                "lines_at_opening": ";".join(operating) or "",
                "n_lines_at_opening": len(operating),
                "is_transfer_at_opening": bool(len(operating) >= 2),
                "is_new_line_opening": bool(has_new),
                "is_extension_opening": bool(has_existing),
                "is_terminal_at_opening": terminal_value,
                "terminal_evidence": terminal_evidence,
                "terminal_degree_current": int(deg_current),
                "terminal_degree_open": int(deg_open),
                "same_month_openings": int(len(same_month_entities)),
                "same_month_new_line_stations": int(same_month_new),
                "treatment_order": event_to_treatment.get(eid),
                "is_treated": eid in event_to_treatment,
            }
        )

    result = pd.DataFrame.from_records(records)
    result["is_treated"] = result["is_treated"].astype(bool)

    treated = result.loc[result["is_treated"]].copy()
    diagnostics: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "n_stations": int(len(result)),
        "n_treated": int(len(treated)),
        "adjacency_rows": int(len(adjacency)),
        "adjacency_rows_with_both_stations_matched": int(adjacency_matched_count),
        "missing_lines": int(result["lines_raw"].isna().sum()),
        "treated": {
            "transfer_at_opening": {
                "rate": round(float(treated["is_transfer_at_opening"].mean()), 4),
                "count": int(treated["is_transfer_at_opening"].sum()),
            },
            "new_line_opening": {
                "rate": round(float(treated["is_new_line_opening"].mean()), 4),
                "count": int(treated["is_new_line_opening"].sum()),
            },
            "extension_opening": {
                "rate": round(float(treated["is_extension_opening"].mean()), 4),
                "count": int(treated["is_extension_opening"].sum()),
            },
                "terminal_at_opening": {
                    "rate": round(
                        float(treated["is_terminal_at_opening"].mean(skipna=True)), 4
                    )
                    if treated["is_terminal_at_opening"].notna().any()
                    else None,
                    "count": int(treated["is_terminal_at_opening"].sum(skipna=True)),
                    "missing": int(treated["is_terminal_at_opening"].isna().sum()),
                    "transfer_ambiguous": int(
                        (treated["terminal_evidence"] == "transfer_degree_ambiguous").sum()
                    ),
                    "no_adjacency_evidence": int(
                        (treated["terminal_evidence"] == "no_adjacency").sum()
                    ),
                },
            "same_month_openings_mean": round(float(treated["same_month_openings"].mean()), 2),
        },
        "all_stations": {
            "transfer_at_opening_rate": round(
                float(result["is_transfer_at_opening"].mean()), 4
            ),
            "new_line_opening_rate": round(float(result["is_new_line_opening"].mean()), 4),
            "extension_opening_rate": round(float(result["is_extension_opening"].mean()), 4),
        },
        "examples": {
            "treated_transfer": treated.loc[
                treated["is_transfer_at_opening"],
                ["city_key", "canonical_station_name", "lines_raw", "lines_at_opening"],
            ]
            .head(5)
            .astype(str)
            .to_dict(orient="records"),
            "treated_new_line": treated.loc[
                treated["is_new_line_opening"],
                ["city_key", "canonical_station_name", "opening_date", "lines_raw"],
            ]
            .head(5)
            .astype(str)
            .to_dict(orient="records"),
            "treated_terminal": treated.loc[
                treated["is_terminal_at_opening"].astype(bool),
                ["city_key", "canonical_station_name", "lines_at_opening"],
            ]
            .head(5)
            .astype(str)
            .to_dict(orient="records"),
        },
    }
    return result, diagnostics


def main() -> int:
    events, adjacency, treatments = load_frames()
    attributes, diagnostics = compute_station_attributes(events, adjacency, treatments)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    attributes.to_parquet(OUT_PATH, index=False, compression="zstd")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_PATH.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(attributes)} station rows to {OUT_PATH.relative_to(ROOT)} "
        f"({diagnostics['n_treated']} treated)"
    )
    print(json.dumps(diagnostics["treated"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
