"""English presentation figures from the existing frozen-data plotting workflow.

Exact reuse: preserve observations, estimates, intervals, support, transforms,
markers and panel order. Change labels, typography and output directory only.
Conceptual diagrams remain distinct from quantitative experiment panels.
"""
import importlib.util
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

SCRIPT = Path(__file__).with_name("plot_project_intro_figures.py")
spec = importlib.util.spec_from_file_location("intro_frozen_figures", SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

FONT = "Times New Roman"
font_path = font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
assert font_manager.FontProperties(fname=font_path).get_name() == FONT
plt.rcParams.update({
    "font.family": "serif", "font.serif": [FONT], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 12,
})
base.OUT = base.OUT / "en"

TRANSLATIONS = {
    "站点开通与网格面板\n5,048 事件 · 44 城市": "Station openings and grids\n5,048 events / 44 cities",
    "处理前特征 X\n训练城市内标准化": "Pre-treatment features X\nTraining-side standardization",
    "因果响应估计\n诊断 · 权重 · 冻结": "Causal response estimates\nDiagnostics / weights / freeze",
    "表格编码器\n归一化嵌入 e": "Tabular encoder\nNormalized embedding e",
    "响应监督\n形状相似度 + 预测": "Response supervision\nShape similarity + prediction",
    "未见城市评估\n相对 raw / 外观基线": "Held-out city evaluation\nRaw / appearance baselines",
    "训练时计算损失": "Training loss",
    "推断仅需要 X": "Inference uses X",
    "同城匹配": "Within-city\nmatching",
    "同城 GSC": "Within-city GSC",
    "同城 MC": "Within-city MC",
    "跨城匹配": "Cross-city\nmatching",
    "跨城 GSC": "Cross-city GSC",
    "跨城 MC": "Cross-city MC",
    "跳过": "Skip",
    "任一步可用即保留\n否则继续下一步": "Keep the first supported route;\notherwise continue",
    "每个网格 × 结果族分配一种方法；不是三方法集成": "One assigned method per grid and outcome family",
    "处理前表格 X\n年度特征 · 1–3 年 lag": "Pre-treatment X\nAnnual features\n1-3 year lags",
    "MLP 编码器\n256 → 128": "MLP encoder\n256 to 128",
    "投影头\n128 维 · L2 归一化": "Projection head\n128 dimensions\nL2 normalized",
    "嵌入检索 / 预测头\n按结果族输出": "Embedding / prediction heads\nOutputs by outcome family",
    "冻结响应 r + 掩码\n设计权重 W + 估计 SE": "Frozen responses r + masks\nDesign weights W + estimated SE",
    "成对响应相似度\n对齐损失 + 距离损失": "Pairwise response similarity\nAlignment + distance losses",
    "逐 cell 预测损失\n使用有效标签及权重": "Cell-level prediction loss\nValid labels and weights",
    "联合训练：0.25 对齐 + 0.25 距离\n+ 0.50 预测": "Joint training: 0.25 alignment\n+ 0.25 distance + 0.50 prediction",
    "城市 CV 最小对照\nraw → ridge / 小型回归": "City-level CV controls\nRaw / ridge / small regressors",
    "城市外预测\n是否出现稳定增量？": "Held-out city prediction\nStable improvement?",
    "有：检查表示是否保留该信号\n直接回归 vs 表示 vs raw+响应块": "Gain: test signal preservation\nRegression / embedding / raw + response",
    "未发现：检查支持、标签与特征\n补充时点安全变量，再评估边界": "No observed gain: examine support and targets\nAdd pre-treatment features; reassess",
    "有": "Gain",
    "未发现": "No observed gain",
    "未发现增量 ≠ 已证明无信号；最终结论依据城市外配对比较": "Assess gains through paired comparisons in held-out cities",
    "无数据": "Unavailable",
    "支持数量（对数色阶）": "Support count (log color scale)",
    "同城支持数量（对数坐标；不同方法口径见正文）": "Within-city support (log scale; method-specific definitions)",
    "匹配：处理前系数联合 Wald": "Matching: joint pre-treatment Wald test",
    "GSC / MC：城市平均 pre-gap": "GSC / MC: city mean pre-treatment gap",
    "p 值（0.05 为参考线）": "p-value (reference line: 0.05)",
    "住房": "Housing",
    "VIIRS 夜间灯光": "VIIRS nighttime lights",
    "POI（count）": "POI count",
    "人口": "Population",
    "匹配": "Matching",
    "数量（log）": "Count (log)",
    "业态熵": "Category entropy",
    "商业占比": "Commercial share",
    "交通可达": "Transport access",
    "城市平均 pre-gap 检验 p 值": "p-value: city mean pre-treatment gap",
    "无可用曲线": "Curve unavailable",
    "相对开通时间（月）": "Months relative to opening",
    "相对开通时间（年）": "Years relative to opening",
    "估计效应（结果变量变换尺度）": "Estimated effect (transformed outcome)",
    "原始特征": "Raw features",
    "随机近邻": "Random neighbors",
    "外观自编码器": "Appearance autoencoder",
    "监督表示 · 默认": "Supervised: default",
    "监督表示 · 关闭队列": "Supervised: queue off",
    "nn_corr@5（越高越好）": "nn_corr@5 (higher is better)",
    "形状检索 ↑\n原始特征 vs 随机近邻": "Shape retrieval (higher is better)\nRaw vs random neighbors",
    "幅度误差 ↓\nridge vs 训练均值": "Magnitude error (lower is better)\nRidge vs training mean",
    "方向准确率 ↑\nridge vs 训练多数符号": "Direction accuracy (higher is better)\nRidge vs training majority sign",
    "准确率（%）": "Accuracy (%)",
    "开通年份": "Opening year",
    "站点开通事件数": "Station opening events",
    "设计不可靠度 D（序数分数）": "Design unreliability D (ordinal score)",
    "设计分量 exp(−γD)": "Design component exp(-γD)",
    "归一化精度 p": "Normalized precision p",
    "精度分量 p^α": "Precision component p^α",
}


def translate(text):
    if text in TRANSLATIONS:
        return TRANSLATIONS[text]
    if text.startswith("城市簇："):
        return text.replace("城市簇：", "City clusters: ").replace("–", "-")
    if text.startswith("空心点：h="):
        reference = re.search(r"h=(-?\d+)", text).group(1)
        return f"Open circle: reference h={reference}, normalized to zero (no CI)"
    for method in ("匹配", "GSC", "MC"):
        for scope, translated in (("同城", "Within-city"), ("跨城", "Cross-city")):
            if text == f"{method}\n{scope}":
                return f"{'Matching' if method == '匹配' else method}\n{translated}"
    if re.search(r"[\u3400-\u9fff]", text):
        raise ValueError(f"Untranslated figure text: {text!r}")
    return text


font_audit = {}


def save_english(fig, name):
    # Materialize tick labels before auditing all text artists.
    for ax in fig.axes:
        for direction in ("x", "y"):
            labels = getattr(ax, f"get_{direction}ticklabels")()
            if any(re.search(r"[\u3400-\u9fff]", label.get_text()) for label in labels):
                ticks = getattr(ax, f"get_{direction}ticks")()
                getattr(ax, f"set_{direction}ticks")(
                    ticks, [translate(label.get_text()) for label in labels])
    labels = []
    for item in fig.findobj(Text):
        item.set_text(translate(item.get_text()))
        item.set_fontfamily(FONT)
        labels.append(item.get_text())
    font_audit[name] = {"font_family": FONT, "text_labels": labels, "cjk_labels": 0}
    fig.savefig(base.OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(base.OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(base.OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


base.save = save_english


def main():
    base.main()
    (base.OUT / "english_typography_audit.json").write_text(
        json.dumps(font_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"English figure text checked; {FONT} resolved and embedded in PDF exports.")


if __name__ == "__main__":
    main()
