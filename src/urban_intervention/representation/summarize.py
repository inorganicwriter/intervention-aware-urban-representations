"""Summarise trained runs into a comparison table (paper Table 1 draft).

Scans one or more training output directories for ``runs.jsonl`` records
(written by every ``train_representation`` call) and emits a flat CSV and a
markdown table with one row per run: dataset, config hash, headline test
metrics, and each chance/appearance baseline's ``nn_corr@k``.

Usage:
    urban-summarize-runs outputs/representation/main outputs/representation/seed_1 ... \\
        --output outputs/representation/summary

Also exposed as the ``urban-summarize-runs`` console script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HEADLINE_COLUMNS = [
    "best_val_loss",
    "test_nn_corr@k",
    "test_permutation_p",
    "test_probe_emb_rmse",
    "test_probe_raw_rmse",
]


def _run_label(path: Path) -> str:
    """A short, stable label for one run directory."""
    if path.name.startswith("seed_"):
        return f"{path.parent.name}::{path.name}"
    return path.name


def collect_runs(directories: list[Path]) -> list[dict[str, object]]:
    """One row per run directory, from the latest runs.jsonl record."""
    rows: list[dict[str, object]] = []
    for directory in directories:
        runs_path = directory / "runs.jsonl"
        if not runs_path.is_file():
            print(f"[skip] no runs.jsonl in {directory}", file=sys.stderr)
            continue
        records = [
            json.loads(line)
            for line in runs_path.read_text(encoding="utf-8").strip().splitlines()
            if line.strip()
        ]
        if not records:
            print(f"[skip] empty runs.jsonl in {directory}", file=sys.stderr)
            continue
        latest = records[-1]
        row: dict[str, object] = {
            "run": _run_label(directory),
            "dataset_id": latest.get("dataset_id", ""),
            "config_sha256": str(latest.get("config_sha256", ""))[:12],
            "created_utc": latest.get("created_utc", ""),
        }
        row.update(_headline_metrics(directory, latest))
        rows.append(row)
    return rows


def _headline_metrics(directory: Path, record: dict[str, object]) -> dict[str, object]:
    """Headline metrics from the run record, with the permutation p from the
    evaluation report (the record itself carries only nn_corr@k)."""
    metrics: dict[str, object] = {}
    for column in HEADLINE_COLUMNS:
        value = record.get(column)
        if isinstance(value, (int, float)):
            metrics[column] = round(float(value), 6)
    report_path = directory / "evaluation_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
        test_entry = report.get("test")
        if isinstance(test_entry, dict):
            permutation = test_entry.get("permutation")
            if isinstance(permutation, dict) and isinstance(
                permutation.get("p_value"), (int, float)
            ):
                metrics["test_permutation_p"] = round(float(permutation["p_value"]), 4)
            probe = test_entry.get("probe")
            if isinstance(probe, dict):
                for key, probe_name in (
                    ("test_probe_emb_rmse", "embeddings"),
                    ("test_probe_raw_rmse", "raw_features"),
                ):
                    probe_entry = probe.get(probe_name)
                    if isinstance(probe_entry, dict) and isinstance(
                        probe_entry.get("rmse_overall"), (int, float)
                    ):
                        metrics[key] = round(float(probe_entry["rmse_overall"]), 6)
    baselines = record.get("baseline_nn_corr@k")
    if isinstance(baselines, dict):
        for name, value in baselines.items():
            if isinstance(value, (int, float)):
                metrics[f"baseline_{name}_nn_corr@k"] = round(float(value), 6)
    return metrics


def render_markdown(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    columns = [column for column in rows[0] if column != "created_utc"]
    lines = ["# Run summary", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(column, "") if row.get(column) is not None else "") for column in columns)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Training output directories")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/representation/summary"),
        help="Output prefix; writes <prefix>.csv and <prefix>.md",
    )
    args = parser.parse_args()
    rows = collect_runs(args.run_dirs)
    if not rows:
        print("No runs.jsonl records found in the given directories", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)  # type: ignore[arg-type]
    args.output.with_suffix(".md").write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote {args.output.with_suffix('.csv')} and {args.output.with_suffix('.md')} "
          f"({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
