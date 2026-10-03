"""Generate the project-introduction figures from local release artifacts.

Outputs (PDF, English labels to avoid CJK font issues):
  intro_pipeline.pdf      项目流程
  intro_support.pdf       逐族标签支持规模
  intro_pretrend.pdf      城市聚类预趋势 p 值
  intro_asset_map.pdf     availability 状态矩阵
  intro_weight_curve.pdf  可靠性权重分量示意
  intro_evidence.pdf      真实数据证据对照
  intro_decision.pdf      最小对照决策树
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "outputs" / "event_study_final"
OUT = ROOT / "docs" / "presentations" / "figures"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _save(figure: plt.Figure, name: str) -> None:
    figure.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def pipeline_figure() -> None:
    figure, axis = plt.subplots(figsize=(12.4, 4.6))
    axis.set_xlim(0, 12.4)
    axis.set_ylim(0, 4.6)
    axis.axis("off")
    boxes = [
        ("5,048 station openings\n40+ cities, staggered timing", 0.15, 2.55),
        ("Pre-treatment features\nhousing / POI / VIIRS / population / S2\nlag 1-3, train-city scaling", 3.25, 2.55),
        ("Causal estimators\nmatched event study · GSC\nmatrix completion", 6.35, 2.55),
        ("Identification diagnostics\npre-trend · holdout · CV MSPE\nsupport counts", 9.45, 2.55),
        ("Reliability weights\nW = A · exp(-gamma D) · precision^alpha\nper method x family", 9.45, 0.35),
        ("Frozen release\nmanifest + SHA256 + waiver\nreleases/production", 6.35, 0.35),
        ("Response-aligned representation\nsoft-label contrastive + prediction head\nper outcome family", 3.25, 0.35),
        ("Retrieval evaluation\nnn_corr@k vs raw / appearance baselines\ncity-level CV", 0.15, 0.35),
    ]
    for text, x, y in boxes:
        box = FancyBboxPatch(
            (x, y), 2.8, 1.45, boxstyle="round,pad=0.06", linewidth=0.9,
            edgecolor="#3b6ea5", facecolor="#eaf1f8",
        )
        axis.add_patch(box)
        axis.text(x + 1.4, y + 0.72, text, ha="center", va="center", fontsize=7.6)
    arrows = [
        ((2.95, 3.28), (3.25, 3.28)),
        ((6.05, 3.28), (6.35, 3.28)),
        ((9.15, 3.28), (9.45, 3.28)),
        ((10.85, 2.55), (10.85, 1.8)),
        ((9.45, 1.08), (9.15, 1.08)),
        ((6.35, 1.08), (6.05, 1.08)),
        ((3.25, 1.08), (2.95, 1.08)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        axis.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                linewidth=0.9, color="#3b6ea5",
            )
        )
    axis.text(6.2, 4.35, "Causal response pipeline and response-aligned representation learning",
              ha="center", va="center", fontsize=9.5)
    _save(figure, "intro_pipeline")


def support_figure() -> None:
    availability = pd.read_csv(RELEASE / "availability.csv", encoding="utf-8-sig")
    families = ["housing", "viirs", "poi", "population"]
    series = {
        "matched same-city": (
            availability[availability["method"].eq("matching") & availability["donor_scope"].eq("same_city")]
            .set_index("outcome_family")["availability_treated_events"]
        ),
        "matched cross-city": (
            availability[availability["method"].eq("matching") & availability["donor_scope"].eq("all_city_standardized")]
            .set_index("outcome_family")["availability_treated_events"]
        ),
        "GSC same-city": (
            availability[availability["method"].eq("gsc") & availability["donor_scope"].eq("same_city")]
            .set_index("outcome_family")["max_supported_treated_grids"]
        ),
        "MC same-city": (
            availability[availability["method"].eq("mc") & availability["donor_scope"].eq("same_city")]
            .set_index("outcome_family")["max_supported_treated_grids"]
        ),
    }
    figure, axis = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(len(families))
    width = 0.2
    colors = ["#3b6ea5", "#8fb4d9", "#e39a4a", "#7a9e6b"]
    for index, (label, values) in enumerate(series.items()):
        heights = [float(values.get(family, np.nan)) for family in families]
        positions = x + (index - 1.5) * width
        axis.bar(positions, heights, width=width, label=label, color=colors[index], edgecolor="white", linewidth=0.4)
        for position, height in zip(positions, heights, strict=True):
            if np.isfinite(height):
                axis.text(position, height * 1.12, f"{int(height)}", ha="center", va="bottom", fontsize=6.4)
    axis.set_yscale("log")
    axis.set_ylim(3, 9000)
    axis.set_xticks(x)
    axis.set_xticklabels(["housing", "VIIRS", "POI", "population"])
    axis.set_ylabel("labeled grids (log scale)")
    axis.set_title("Same-city supervision scale by estimator", fontsize=9)
    axis.legend(fontsize=6.6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(figure, "intro_support")


def pretrend_figure() -> None:
    pretrend = pd.read_csv(RELEASE / "pretrend_summary.csv", encoding="utf-8-sig")
    city = pretrend[pretrend["cluster"].eq("city") & pretrend["donor_scope"].eq("same_city")].copy()
    families = ["housing", "viirs", "poi", "population"]
    methods = [("matching", "#3b6ea5"), ("gsc", "#e39a4a"), ("mc", "#7a9e6b")]
    figure, axis = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(families))
    width = 0.26
    for index, (method, color) in enumerate(methods):
        heights = []
        for family in families:
            rows = city[city["outcome_family"].eq(family) & city["method"].eq(method)]
            heights.append(float(rows["p_value"].iloc[0]) if not rows.empty else np.nan)
        positions = x + (index - 1) * width
        axis.bar(positions, heights, width=width, color=color, label=method, edgecolor="white", linewidth=0.4)
        for position, height in zip(positions, heights, strict=True):
            if np.isfinite(height):
                axis.text(position, height + 0.012, f"{height:.3f}", ha="center", va="bottom", fontsize=6.2, rotation=90)
    axis.axhline(0.05, color="#b02a2a", linestyle="--", linewidth=0.9)
    axis.axhline(0.10, color="#d98a2b", linestyle=":", linewidth=0.9)
    axis.text(3.42, 0.052, "0.05", fontsize=6.4, color="#b02a2a")
    axis.text(3.42, 0.102, "0.10", fontsize=6.4, color="#d98a2b")
    axis.set_ylim(0, 0.55)
    axis.set_xticks(x)
    axis.set_xticklabels(["housing", "VIIRS", "POI", "population"])
    axis.set_ylabel("pre-trend test p-value (city cluster)")
    axis.set_title("Same-city clean-pre tests by estimator", fontsize=9)
    axis.legend(fontsize=6.6, frameon=False, ncol=3, loc="upper right")
    _save(figure, "intro_pretrend")


def asset_map_figure() -> None:
    availability = pd.read_csv(RELEASE / "availability.csv", encoding="utf-8-sig")
    families = ["housing", "viirs", "poi", "population"]
    columns = [
        ("matching", "same_city", "matched\nsame"),
        ("matching", "all_city_standardized", "matched\ncross"),
        ("gsc", "same_city", "GSC\nsame"),
        ("gsc", "all_city_standardized", "GSC\ncross"),
        ("mc", "same_city", "MC\nsame"),
        ("mc", "all_city_standardized", "MC\ncross"),
    ]
    values = np.full((len(families), len(columns)), np.nan)
    for row_index, family in enumerate(families):
        for column_index, (method, scope, _) in enumerate(columns):
            rows = availability[
                availability["outcome_family"].eq(family)
                & availability["method"].eq(method)
                & availability["donor_scope"].eq(scope)
                & availability["status"].eq("available")
            ]
            if rows.empty:
                continue
            count = rows["availability_treated_events"].iloc[0]
            if not np.isfinite(count):
                count = rows["max_supported_treated_grids"].iloc[0]
            values[row_index, column_index] = float(count) if np.isfinite(count) else np.nan
    figure, axis = plt.subplots(figsize=(8.6, 3.4))
    image = axis.imshow(np.log10(np.nan_to_num(values, nan=0.0) + 1), cmap="Blues", aspect="auto", vmin=0, vmax=3.5)
    for row_index in range(len(families)):
        for column_index in range(len(columns)):
            value = values[row_index, column_index]
            if not np.isfinite(value):
                axis.text(column_index, row_index, "no_data", ha="center", va="center",
                          fontsize=7, color="#8a8a8a")
                continue
            axis.text(column_index, row_index, f"{int(value)}", ha="center", va="center",
                      fontsize=7, color="#333333" if value < 800 else "white")
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels([label for _, _, label in columns], fontsize=7)
    axis.set_yticks(range(len(families)))
    axis.set_yticklabels(["housing", "VIIRS", "POI", "population"], fontsize=7.5)
    axis.set_title("Availability map: labeled treated grids (log color)", fontsize=9)
    figure.colorbar(image, ax=axis, shrink=0.8, label="log10(grids + 1)")
    _save(figure, "intro_asset_map")


def weight_curve_figure() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.0))
    d = np.linspace(0, 1, 200)
    for gamma, style in ((1.0, "-"), (0.5, "--")):
        axes[0].plot(d, np.exp(-gamma * d), style, color="#3b6ea5", linewidth=1.2,
                     label=f"gamma={gamma:g}")
    axes[0].set_xlabel("design unreliability D (percentile)")
    axes[0].set_ylabel("exp(-gamma D)")
    axes[0].set_title("Design decay", fontsize=8.5)
    axes[0].legend(fontsize=6.5, frameon=False)
    p = np.linspace(0.05, 2.5, 200)
    for alpha, style in ((0.0, "--"), (0.5, "-")):
        axes[1].plot(p, np.power(p, alpha), style, color="#e39a4a", linewidth=1.2,
                     label=f"alpha={alpha:g}")
    axes[1].set_xlabel("normalized precision p")
    axes[1].set_ylabel("p^alpha")
    axes[1].set_title("Precision tempering", fontsize=8.5)
    axes[1].legend(fontsize=6.5, frameon=False)
    _save(figure, "intro_weight_curve")


def evidence_figure() -> None:
    families = ["VIIRS (test, n=205)", "POI (test, n=220)"]
    raws = [0.0036, 0.1089]
    chance = [0.0031, 0.0342]
    autoencoder_mean = [0.0078, 0.1113]
    autoencoder_low = [0.0041, 0.1072]
    autoencoder_high = [0.0149, 0.1193]
    model_mean = [0.0150, 0.0955]
    model_sd = [0.0067, 0.0025]
    model_q0_mean = [0.0250, 0.1006]
    model_q0_sd = [0.0184, 0.0174]
    figure, axis = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(families))
    width = 0.16
    axis.bar(x - 1.5 * width, raws, width, label="raw features", color="#9aa3ab", edgecolor="white", linewidth=0.4)
    axis.bar(x - 0.5 * width, autoencoder_mean, width, label="appearance autoencoder",
             color="#e39a4a", edgecolor="white", linewidth=0.4)
    axis.errorbar(x - 0.5 * width, autoencoder_mean,
                  yerr=[np.array(autoencoder_mean) - np.array(autoencoder_low),
                        np.array(autoencoder_high) - np.array(autoencoder_mean)],
                  fmt="none", ecolor="#8a5a1f", elinewidth=0.8, capsize=2)
    axis.bar(x + 0.5 * width, model_mean, width, label="supervised model (3 seeds)",
             color="#3b6ea5", edgecolor="white", linewidth=0.4)
    axis.errorbar(x + 0.5 * width, model_mean, yerr=model_sd, fmt="none",
                  ecolor="#1d3d61", elinewidth=0.8, capsize=2)
    axis.bar(x + 1.5 * width, model_q0_mean, width, label="supervised model, queue off",
             color="#6b8fb5", edgecolor="white", linewidth=0.4)
    axis.errorbar(x + 1.5 * width, model_q0_mean, yerr=model_q0_sd, fmt="none",
                  ecolor="#345a80", elinewidth=0.8, capsize=2)
    axis.scatter(x, chance, marker="_", s=120, color="#b02a2a", label="random-neighbour baseline", zorder=5)
    axis.set_xticks(x)
    axis.set_xticklabels(families)
    axis.set_ylabel("nn_corr@k (response retrieval)")
    axis.set_title("Real-data evidence: no stable gain over raw/appearance", fontsize=9)
    axis.legend(fontsize=6.4, frameon=False, ncol=2)
    _save(figure, "intro_evidence")


def decision_figure() -> None:
    figure, axis = plt.subplots(figsize=(10.4, 4.2))
    axis.set_xlim(0, 10.4)
    axis.set_ylim(0, 4.2)
    axis.axis("off")
    boxes = [
        ("City-level CV control:\nridge / kNN vs refit mean baseline", 0.15, 1.55, 2.3, 1.0, "#eaf1f8", "#3b6ea5"),
        ("Stable out-of-city\nprediction gain?", 2.85, 1.55, 2.2, 1.0, "#fdf3e3", "#e39a4a"),
        ("Representation still\nbelow raw geometry?", 5.85, 2.65, 2.2, 1.0, "#fdf3e3", "#e39a4a"),
        ("Fix the objective:\nraw geometry + compact\nresponse block (lambda=0 allowed)", 8.5, 2.6, 1.75, 1.1, "#e8f1e8", "#7a9e6b"),
        ("Run the multi-seed\nmatrix with the\ncurrent recipe", 5.85, 1.05, 2.2, 1.0, "#e8f1e8", "#7a9e6b"),
        ("Features lack signal:\nadd pre-treatment info\n(station/network, time-safe)\nor reframe the paper", 2.85, 0.2, 2.7, 1.1, "#f3e8ee", "#a05a7a"),
    ]
    for text, x, y, width, height, face, edge in boxes:
        axis.add_patch(
            FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06",
                           linewidth=0.9, edgecolor=edge, facecolor=face)
        )
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=7.4)
    arrows = [
        ((2.45, 2.05), (2.85, 2.05)),
        ((5.05, 2.3), (5.85, 3.1)),
        ((5.05, 1.8), (5.85, 0.8)),
        ((8.05, 3.15), (8.5, 3.15)),
        ((6.95, 2.65), (6.95, 2.05)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        axis.add_patch(
            FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                            linewidth=0.9, color="#666666")
        )
    axis.text(5.32, 2.72, "yes", fontsize=6.8, color="#555555")
    axis.text(5.32, 1.28, "no", fontsize=6.8, color="#555555")
    axis.text(8.22, 3.28, "yes", fontsize=6.8, color="#555555")
    axis.text(7.04, 2.33, "no", fontsize=6.8, color="#555555")
    axis.text(5.2, 3.98, "Decision tree after the reviewed control experiment",
              ha="center", va="center", fontsize=9)
    _save(figure, "intro_decision")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline_figure()
    support_figure()
    pretrend_figure()
    asset_map_figure()
    weight_curve_figure()
    evidence_figure()
    decision_figure()
    print("Wrote figures to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
