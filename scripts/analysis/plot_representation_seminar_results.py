"""Benchmark and ablation plots from reported server representation summaries.

Contract: compare all supplied methods within each family, then show the raw-
relative gains of both supervised variants. Quantitative grids; Times New Roman;
vector PDF/SVG and 300 dpi PNG. No embedding points, seed values or CI inferred.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/presentations/data/representation_history.csv"
OUT = ROOT / "docs/presentations/figures/en"
FONT = "Times New Roman"
font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
plt.rcParams.update({"font.family": "serif", "font.serif": [FONT], "font.size": 16,
                     "axes.titlesize": 20, "axes.labelsize": 16, "axes.linewidth": .8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False})
COLORS = {"raw": "#81919D", "chance": "#BBC5CC", "autoencoder": "#D28436",
          "supervised": "#234E70", "queue_off": "#168A85"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def benchmark(data):
    models = ["raw", "chance", "autoencoder", "supervised", "queue_off"]
    names = ["Raw\nfeatures", "Random\nneighbors", "Appearance\nAE",
             "Supervised\n(default)", "Supervised\n(queue off)"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), layout="constrained")
    for ax, family in zip(axes, ("viirs", "poi"), strict=False):
        rows = data[data.family.eq(family)].set_index("model")
        for x, model in enumerate(models):
            row = rows.loc[model]
            value = float(row["mean"])
            ax.bar(x, value, width=.63, color=COLORS[model], zorder=3)
            high = value
            if row.error_kind == "seed_sd":
                assert row.n_seeds == 3 and row.error_low == row.error_high
                error = [[float(row.error_low)], [float(row.error_high)]]
                high += float(row.error_high)
            elif row.error_kind == "range":
                assert row.error_low <= value <= row.error_high
                error = [[value - float(row.error_low)], [float(row.error_high) - value]]
                high = float(row.error_high)
            else:
                assert row.error_kind == "none"
                error = None
            if error is not None:
                ax.errorbar(x, value, yerr=error, fmt="none", color="#344655",
                            capsize=4, linewidth=1.2, zorder=4)
            offset = .0015 if family == "viirs" else .004
            ax.text(x, high + offset, f"{value:.4f}", ha="center", va="bottom", fontsize=15)
        ax.set_xticks(np.arange(len(models)), names, fontsize=14)
        ax.set_ylim(0, .060 if family == "viirs" else .145)
        ax.set_title(f"{family.upper()} | test n = {int(rows.test_n.iloc[0])}", pad=12)
        ax.set_ylabel("nn_corr@5 (higher is better)")
        ax.grid(axis="y", alpha=.14, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=7)
    save(fig, "intro_representation_benchmark")


def gains(data):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2), layout="constrained")
    records = []
    for ax, family in zip(axes, ("viirs", "poi"), strict=False):
        rows = data[data.family.eq(family)].set_index("model")
        raw = float(rows.loc["raw", "mean"])
        ax.axvspan(0, .046, color="#EEF5F7", zorder=0)
        ax.axvline(0, color="#657786", linestyle="--", linewidth=1)
        for y, model in zip((1, 0), ("supervised", "queue_off"), strict=False):
            row = rows.loc[model]
            delta = float(row["mean"]) - raw
            sd = float(row.error_low)
            ax.errorbar(delta, y, xerr=sd, fmt="o" if model == "supervised" else "s",
                        color=COLORS[model], markersize=9, capsize=5, linewidth=2, zorder=3)
            ax.annotate(f"{delta:+.4f}",(delta,y),(0,15),textcoords="offset points",
                        ha="center",va="bottom",fontsize=18,color=COLORS[model])
            records.append({"family": family, "model": model, "raw_mean": raw,
                            "mean_gain": delta, "seed_sd": sd, "n_seeds": 3})
        ax.set(xlim=(-.034, .046), ylim=(-.5, 1.7),
               xlabel="Change in nn_corr@5 vs raw features",
               title=f"{family.upper()} | raw = {raw:.4f}")
        ax.set_yticks([1, 0], ["Default", "Queue off"])
        ax.set_xticks([-.02, 0, .02, .04], ["-0.02", "0", "+0.02", "+0.04"])
        ax.grid(axis="x", alpha=.12)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.text(.98,.97,"Higher than raw",transform=ax.transAxes,ha="right",va="top",
                fontsize=13,color="#526475")
    save(fig, "intro_representation_gains")
    pd.DataFrame(records).to_csv(OUT / "intro_representation_gains.csv", index=False)


def main():
    data = pd.read_csv(SOURCE)
    assert len(data) == 10 and not data.duplicated(["family", "model"]).any()
    assert np.isfinite(data["mean"]).all()
    OUT.mkdir(parents=True, exist_ok=True)
    benchmark(data)
    gains(data)
    audit = {"source": "Reported server summaries", "input_rows": 10,
             "benchmark_rows": 10, "gain_rows": 4,
             "gain_selection": "Both supervised variants in each family; raw is the fixed reference",
             "gain_definition": "supervised mean minus raw mean",
             "uncertainty": "Supervised: three-seed SD; appearance AE: reported range; neither is CI",
             "gain_sd": "Subtracting fixed raw score leaves reported supervised seed SD unchanged",
             "benchmark_axes": "Independent y-scales, zero baseline",
             "gain_axes": "Shared absolute x-scale with zero reference",
             "font": FONT, "units": "Correlation score; gains are not percentage points",
             "protocol": "Earlier city split; VIIRS test n=205, POI test n=220",
             "embedding_projection": "Unavailable locally; no synthetic embedding points drawn"}
    (OUT / "intro_representation_results_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    print("Exported benchmark and raw-relative ablation plots from all reported methods.")


if __name__ == "__main__":
    main()
