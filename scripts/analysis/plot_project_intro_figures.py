"""Chinese presentation assets from frozen releases; no estimation or training.
Historical retrieval values are a prior server summary, not a local rerun.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "outputs/event_study_final"
OUT = ROOT / "docs/presentations/figures"
FAMILIES = ["housing", "viirs", "poi", "population"]
LABELS = ["住房", "VIIRS 夜间灯光", "POI", "人口"]
CANONICAL = dict(zip(FAMILIES, ["housing_log_price", "viirs_avg_asinh", "poi_count_log", "population_log"], strict=False))
METHODS = {"matching": "匹配", "gsc": "GSC", "mc": "MC"}
COLORS = {"matching": "#234E70", "gsc": "#168A85", "mc": "#D28436"}
AUDIT = {}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False})

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def record(name, source, data, selected, **details):
    AUDIT[name] = {"source": source, "input_rows": len(data), "selected_rows": len(selected), **details}

def box(ax, x, y, w, h, label, color="#234E70"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.025",edgecolor=color,facecolor="#EEF3F7",linewidth=1.2))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=12,color="#16324A")

def arrow(ax, a, b, label=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=15,linewidth=1.3,color="#637381"))
    if label:
        ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha="center",fontsize=11,
                bbox=dict(facecolor="white",edgecolor="none",pad=1))

def canvas(h=4.4):
    fig, ax = plt.subplots(figsize=(12,h))
    ax.set(xlim=(0,12),ylim=(0,h)); ax.axis("off")
    return fig, ax

def diagrams():
    fig, ax = canvas()
    for x,y,w,label in [
        (.1,1.75,2.3,"站点开通与网格面板\n5,048 事件 · 44 城市"),
        (3,3,2.4,"处理前特征 X\n训练城市内标准化"),
        (3,.3,2.4,"因果响应估计\n诊断 · 权重 · 冻结"),
        (6,3,2.4,"表格编码器\n归一化嵌入 e"),
        (6,.3,2.4,"响应监督\n形状相似度 + 预测"),
        (9.2,1.75,2.6,"未见城市评估\n相对 raw / 外观基线")]:
        box(ax,x,y,w,1,label)
    for a,b in [((2.4,2.55),(3,3.4)),((2.4,1.95),(3,.9)),((5.4,3.5),(6,3.5)),
                ((5.4,.8),(6,.8)),((8.4,3.35),(9.2,2.45))]:
        arrow(ax,a,b)
    arrow(ax,(7.2,3),(7.2,1.3),"训练时计算损失")
    ax.text(10.5,.85,"推断仅需要 X",ha="center",color="#168A85",fontsize=13)
    save(fig,"intro_pipeline")
    fig, ax = canvas(3.4)
    seq = [("同城匹配",.1,2),("同城 GSC",3.1,2),("同城 MC",6.1,2),
           ("跨城匹配",9.1,2),("跨城 GSC",9.1,.3),("跨城 MC",6.1,.3),("跳过",3.1,.3)]
    for label,x,y in seq:
        box(ax,x,y,2.6,.9,label)
    for a,b in [((2.7,2.45),(3.1,2.45)),((5.7,2.45),(6.1,2.45)),
                ((8.7,2.45),(9.1,2.45)),((10.4,2),(10.4,1.2)),
                ((9.1,.75),(8.7,.75)),((6.1,.75),(5.7,.75))]:
        arrow(ax,a,b)
    ax.text(.2,.75,"任一步可用即保留\n否则继续下一步",fontsize=12,va="center",color="#168A85")
    ax.text(6,3.15,"每个网格 × 结果族分配一种方法；不是三方法集成",ha="center",fontsize=13)
    save(fig,"intro_routing")
    fig, ax = canvas(4.5)
    for x,w,label in [(.1,2.25,"处理前表格 X\n年度特征 · 1–3 年 lag"),
        (2.9,2.2,"MLP 编码器\n256 → 128"),(5.7,2.25,"投影头\n128 维 · L2 归一化"),
        (8.7,3,"嵌入检索 / 预测头\n按结果族输出")]:
        box(ax,x,3.1,w,1.1,label)
    for a,b in [((2.35,3.65),(2.9,3.65)),((5.1,3.65),(5.7,3.65)),((7.95,3.65),(8.7,3.65))]:
        arrow(ax,a,b)
    for x,w,label in [(.1,3.5,"冻结响应 r + 掩码\n设计权重 W + 估计 SE"),
                      (4.5,3.5,"成对响应相似度\n对齐损失 + 距离损失"),
                      (8.7,3,"逐 cell 预测损失\n使用有效标签及权重")]:
        box(ax,x,.35,w,1.25,label,"#168A85")
    for a,b in [((3.6,.98),(4.5,.98)),((7,3.1),(6.4,1.6)),((10.2,3.1),(10.2,1.6))]:
        arrow(ax,a,b)
    ax.plot([1.85,1.85,10.2],[.35,.05,.05],color="#637381",linewidth=1.3)
    arrow(ax,(10.2,.05),(10.2,.35))
    ax.text(3.15,2.3,"联合训练：0.25 对齐 + 0.25 距离\n+ 0.50 预测",ha="center",fontsize=12)
    save(fig,"intro_model")
    fig, ax = canvas(4.5)
    for x,y,w,label in [(.1,1.8,2.8,"城市 CV 最小对照\nraw → ridge / 小型回归"),
        (3.5,1.8,2.9,"城市外预测\n是否出现稳定增量？"),
        (7.25,3,4.5,"有：检查表示是否保留该信号\n直接回归 vs 表示 vs raw+响应块"),
        (7.25,.5,4.5,"未发现：检查支持、标签与特征\n补充时点安全变量，再评估边界")]:
        box(ax,x,y,w,1.05,label)
    arrow(ax,(2.9,2.32),(3.5,2.32))
    arrow(ax,(6.4,2.6),(7.25,3.5),"有")
    arrow(ax,(6.4,2.05),(7.25,1),"未发现")
    ax.text(5.9,.15,"未发现增量 ≠ 已证明无信号；最终结论依据城市外配对比较",ha="center",fontsize=12)
    save(fig,"intro_decision")
    for name in ("intro_pipeline","intro_routing","intro_model","intro_decision"):
        AUDIT[name] = {"type": "conceptual", "quantitative_experiment": False}

def availability_figures():
    data = pd.read_csv(RELEASE / "availability.csv")
    columns = [(m,s) for m in METHODS for s in ("same_city","all_city_standardized")]
    values = np.full((4,6),np.nan)
    for i,f in enumerate(FAMILIES):
        for j,(m,s) in enumerate(columns):
            rows = data[data.outcome_family.eq(f)&data.method.eq(m)&data.donor_scope.eq(s)]
            assert len(rows) == 1
            row = rows.iloc[0]
            if row.status == "available":
                values[i,j] = row["availability_treated_events" if m == "matching" else "max_supported_treated_grids"]
    fig, ax = plt.subplots(figsize=(11,4.1),layout="constrained")
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#F1F3F5")
    im = ax.imshow(np.ma.masked_invalid(values),norm=LogNorm(vmin=1,vmax=3000),cmap=cmap,aspect="auto")
    for (i,j),v in np.ndenumerate(values):
        ax.text(j,i,f"{int(v):,}" if np.isfinite(v) else "无数据",ha="center",va="center",
                color="white" if v > 500 else "#263746",fontsize=14)
    ax.set_xticks(range(6),[f"{METHODS[m]}\n{'同城' if s == 'same_city' else '跨城'}" for m,s in columns])
    ax.set_yticks(range(4),LABELS)
    fig.colorbar(im,ax=ax,shrink=.85,label="支持数量（对数色阶）")
    save(fig,"intro_asset_map")
    record("intro_asset_map","availability.csv",data,data,missing_cells=int(np.isnan(values).sum()),
           fields={"matching":"availability_treated_events","gsc/mc":"max_supported_treated_grids"},
           counts_are_final_training_mask=False)
    fig, ax = plt.subplots(figsize=(10,4.5),layout="constrained")
    for j,m in enumerate(METHODS):
        v = values[:,j*2]; y = np.arange(4)+(j-1)*.19; valid = np.isfinite(v)&(v > 0)
        ax.scatter(v[valid],y[valid],s=80,color=COLORS[m],label=METHODS[m])
        for x,yy in zip(v[valid],y[valid],strict=False):
            ax.annotate(f"{int(x):,}",(x,yy),(6,0),textcoords="offset points",va="center",fontsize=11)
    ax.set(xscale="log",xlim=(3,6500),xlabel="同城支持数量（对数坐标；不同方法口径见正文）")
    ax.set_yticks(range(4),LABELS); ax.invert_yaxis(); ax.grid(axis="x",alpha=.15)
    ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(.5,1.13))
    save(fig,"intro_support")
    record("intro_support","availability.csv",data,data[data.donor_scope.eq("same_city")],missing="住房 GSC")

def pformat(p):
    return f"{p:.1e}" if p < .001 else f"{p:.3f}"

def pretrend_figures():
    data = pd.read_csv(RELEASE / "pretrend_summary.csv")
    sel = data[data.cluster.eq("city")&data.donor_scope.eq("same_city")
               &data.apply(lambda r:r.outcome == CANONICAL[r.outcome_family],axis=1)]
    fig, axes = plt.subplots(1,2,figsize=(11.8,4.2),layout="constrained")
    for ax,methods,title,test in [(axes[0],["matching"],"匹配：处理前系数联合 Wald","joint_wald_pretrend"),
        (axes[1],["gsc","mc"],"GSC / MC：城市平均 pre-gap","cluster_mean_pre_gap_zero")]:
        for j,m in enumerate(methods):
            for i,f in enumerate(FAMILIES):
                rows = sel[sel.method.eq(m)&sel.outcome_family.eq(f)]
                if rows.empty: continue
                assert len(rows) == 1 and rows.iloc[0].test_type == test
                p = float(rows.iloc[0].p_value); yy = i+(j-(len(methods)-1)/2)*.2
                ax.scatter(p,yy,s=65,color=COLORS[m],label=METHODS[m] if i == 1 else None)
                ax.annotate(pformat(p),(p,yy),(-6 if p > .8 else 6,0),textcoords="offset points",
                            ha="right" if p > .8 else "left",va="center",fontsize=10)
        ax.axvline(.05,linestyle="--",color="#B55353",linewidth=1)
        ax.set(xlim=(-.015,1.04),ylim=(3.6,-.6),xlabel="p 值（0.05 为参考线）",title=title)
        ax.set_yticks(range(4),[label+("（count）" if f == "poi" else "") for label,f in zip(LABELS,FAMILIES,strict=False)])
        ax.legend(frameon=False,loc="lower right")
    save(fig,"intro_pretrend")
    record("intro_pretrend","pretrend_summary.csv",data,sel,canonical_outcomes=CANONICAL,tests_separated=True,
           values=sel[["method","outcome","p_value"]].to_dict("records"))
    poi = data[data.cluster.eq("city")&data.donor_scope.eq("same_city")&data.outcome_family.eq("poi")&data.method.isin(["gsc","mc"])]
    names = sorted(poi.outcome.unique())
    preferred = [("count","数量（log）"),("entropy","业态熵"),("commercial","商业占比"),("transport","交通可达")]
    outcomes = [next(n for n in names if token in n) for token,_ in preferred]
    fig, ax = plt.subplots(figsize=(10.5,4.3),layout="constrained")
    for j,m in enumerate(("gsc","mc")):
        for i,o in enumerate(outcomes):
            rows = poi[poi.method.eq(m)&poi.outcome.eq(o)]; assert len(rows) == 1
            p = float(rows.iloc[0].p_value); y = i+(j-.5)*.2
            ax.scatter(p,y,color=COLORS[m],s=70,label=METHODS[m] if i == 0 else None)
            ax.annotate(pformat(p),(p,y),(7,0),textcoords="offset points",va="center",fontsize=11)
    ax.axvline(.05,color="#B55353",linestyle="--",linewidth=1)
    ax.set(xlim=(-.01,.57),ylim=(3.5,-.5),xlabel="城市平均 pre-gap 检验 p 值")
    ax.set_yticks(range(4),[label for _,label in preferred]); ax.legend(frameon=False,loc="upper right")
    save(fig,"intro_pretrend_poi")
    record("intro_pretrend_poi","pretrend_summary.csv",data,poi,all_four_outcomes=True)

def event_figures():
    data = pd.read_csv(RELEASE / "event_study_panel.csv")
    windows = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))["windows"]
    for f in FAMILIES:
        window = windows[f]
        lo, hi = int(window["canonical_pre_min"]), int(window["canonical_post_max"])
        reference = int(window["reference_event_time"])
        sel = data[data.outcome_family.eq(f)&data.outcome.eq(CANONICAL[f])&data.donor_scope.eq("same_city")&data.event_time.between(lo,hi)]
        fig, axes = plt.subplots(1,3,figsize=(12,4.2),layout="constrained")
        added_references = []
        missing_periods = {}
        for ax,m in zip(axes,METHODS,strict=False):
            rows = sel[sel.method.eq(m)].sort_values("event_time")
            ax.axhline(0,color="#85909A",linewidth=.8); ax.axvline(0,color="#85909A",linewidth=.9,linestyle="--")
            ax.axvspan(lo,0,color="#F3F5F7",zorder=-1)
            if rows.empty:
                ax.text(.5,.5,"无可用曲线",transform=ax.transAxes,ha="center",fontsize=14)
            else:
                assert not rows.event_time.duplicated().any()
                times = set(rows.event_time.astype(int))
                missing_periods[m] = [h for h in range(min(times), max(times)+1)
                                      if h not in times and h != reference]
                if m == "matching" and reference not in times:
                    # The released TWFE reference is a known normalization, not missing data.
                    # Add it only for display; never write it into the frozen release or CI.
                    rows = pd.concat([rows, pd.DataFrame({"event_time": [reference],
                        "estimate": [0.0], "event_phase": ["reference"],
                        "ci_lower_city": [np.nan], "ci_upper_city": [np.nan],
                        "n_city_clusters": [np.nan]})], ignore_index=True).sort_values("event_time")
                    added_references.append(m)
                x=rows.event_time.to_numpy(); y=rows.estimate.to_numpy()
                is_reference = rows.event_time.eq(reference).to_numpy() & rows.event_phase.eq("reference").to_numpy()
                assert np.allclose(y[is_reference], 0), "Released reference must be normalized to zero"
                lower=rows.ci_lower_city.to_numpy(); upper=rows.ci_upper_city.to_numpy()
                valid=(~is_reference)&(rows.n_city_clusters.to_numpy() >= 3)&np.isfinite(lower)&np.isfinite(upper)
                assert np.all(lower[valid] <= upper[valid])
                ax.vlines(x[valid],lower[valid],upper[valid],colors=COLORS[m],linewidth=1,alpha=.5)
                ax.scatter(x[valid],y[valid],color=COLORS[m],s=18,zorder=3)
                unsupported = (~valid)&(~is_reference)
                ax.scatter(x[unsupported],y[unsupported],color=COLORS[m],marker="x",s=35,zorder=3)
                ax.scatter(x[is_reference],y[is_reference],facecolors="white",edgecolors=COLORS[m],
                           marker="o",s=42,linewidths=1.4,zorder=4)
                for segment in np.split(np.arange(len(x)),np.where(np.diff(x) != 1)[0]+1):
                    ax.plot(x[segment],y[segment],color=COLORS[m],linewidth=1.3)
                nmin, nmax = int(rows.n_city_clusters.min()), int(rows.n_city_clusters.max())
                cluster_label = str(nmin) if nmin == nmax else f"{nmin}–{nmax}"
                ax.text(.03,.97,f"城市簇：{cluster_label}",
                        transform=ax.transAxes,va="top",fontsize=10,color="#526475")
            ax.set(title=METHODS[m],xlim=(lo-.5,hi+.5),xlabel="相对开通时间（月）" if hi == 24 else "相对开通时间（年）")
            ax.set_xticks([-24,-12,0,12,24] if hi == 24 else [-6,-3,0,3]); ax.grid(axis="y",alpha=.13)
        axes[0].set_ylabel("估计效应（结果变量变换尺度）")
        fig.suptitle(f"空心点：h={reference} 参考期，归一化为 0（非估计值，无误差条）",
                     fontsize=11, color="#526475")
        name=f"intro_{f}_event_study"; save(fig,name)
        record(name,"event_study_panel.csv",data,sel,canonical_outcome=CANONICAL[f],scope="same_city",window=[lo,hi],
               missing_estimate=int(sel.estimate.isna().sum()),n_less_than_three_city_clusters=int((sel.n_city_clusters < 3).sum()),
               interval_fields=["ci_lower_city","ci_upper_city"],separate_y_axes=True,
               reference_event_time=reference,reference_source="manifest.json/windows",
               display_reference_rows_added=added_references,reference_has_ci=False,
               true_missing_periods_within_observed_span=missing_periods,
               specifications=sel[["method","specification","p_value_method"]].drop_duplicates().fillna("").to_dict("records"))

def history_figure():
    data=pd.read_csv(ROOT / "docs/presentations/data/representation_history.csv")
    labels=["原始特征","随机近邻","外观自编码器","监督表示 · 默认","监督表示 · 关闭队列"]
    models=["raw","chance","autoencoder","supervised","queue_off"]
    colors=["#657786","#B55353","#D28436","#234E70","#168A85"]
    fig, axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    for ax,f in zip(axes,("viirs","poi"),strict=False):
        sub=data[data.family.eq(f)].set_index("model")
        for i,(m,color) in enumerate(zip(models,colors,strict=False)):
            row=sub.loc[m]; mean=row["mean"]
            if row.error_kind == "seed_sd": error=[[row.error_low],[row.error_high]]; marker="o"
            elif row.error_kind == "range": error=[[mean-row.error_low],[row.error_high-mean]]; marker="s"
            else: error=None; marker="D"
            ax.errorbar(mean,i,xerr=error,fmt=marker,color=color,markersize=7,capsize=4,linewidth=1.8)
            ax.annotate(f"{mean:.4f}",(mean,i),(0,12),textcoords="offset points",ha="center",fontsize=11)
        ax.axvline(sub.loc["raw","mean"],color=colors[0],linestyle=":",linewidth=1)
        ax.set_yticks(range(5),labels); ax.set_ylim(4.6,-.7)
        ax.set(title=f"{f.upper()} · test n = {int(sub.test_n.iloc[0])}",xlabel="nn_corr@5（越高越好）")
        ax.grid(axis="x",alpha=.15)
    save(fig,"intro_evidence")
    record("intro_evidence","docs/presentations/data/representation_history.csv",data,data,
           provenance="Previously reported server summary; not locally re-extracted",
           errors="square: AE reported range; circles: 3-seed mean +/- SD; neither is CI",corrected_city_cv=False)

def city_controls_figure():
    """Display the supplied server summaries; no new fit or inferred interval."""
    source = "docs/presentations/data/representation_city_controls.csv"
    data = pd.read_csv(ROOT / source)
    assert len(data) == 12 and not data.duplicated(["family", "metric", "method"]).any()
    assert np.isfinite(data.value).all() and data.fold_count.eq(5).all()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    settings = [
        ("shape", "raw", "chance", "形状检索 ↑\n原始特征 vs 随机近邻", "nn_corr@5", (0, .085)),
        ("amplitude", "ridge", "train_mean", "幅度误差 ↓\nridge vs 训练均值", "RMSE", (.32, .50)),
        ("direction", "ridge", "train_majority", "方向准确率 ↑\nridge vs 训练多数符号", "准确率（%）", (45, 65)),
    ]
    for ax, (metric, model, baseline, title, xlabel, limits) in zip(axes, settings, strict=False):
        subset = data[data.metric.eq(metric)].set_index(["family", "method"])
        for family, y in (("viirs", 1), ("poi", 0)):
            estimate = float(subset.loc[(family, model), "value"])
            reference = float(subset.loc[(family, baseline), "value"])
            multiplier = 100 if metric == "direction" else 1
            a, b = estimate * multiplier, reference * multiplier
            ax.plot([a, b], [y, y], color="#B5C1CA", linewidth=2.2, zorder=1)
            ax.scatter(b, y, s=75, marker="s", color="#81919D", zorder=2)
            ax.scatter(a, y, s=85, marker="o", color="#168A85", zorder=3)
            fmt = ".1f" if metric == "direction" else ".4f"
            ax.annotate(format(a, fmt), (a, y), (0, 12), textcoords="offset points",
                        ha="center", fontsize=12, color="#116A66")
            ax.annotate(format(b, fmt), (b, y), (0, -23), textcoords="offset points",
                        ha="center", fontsize=12, color="#526475")
        ax.set(title=title, xlabel=xlabel, xlim=limits, ylim=(-.55, 1.55))
        ax.set_yticks([1, 0], ["VIIRS", "POI"] if ax is axes[0] else ["", ""])
        ax.grid(axis="x", alpha=.15)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    save(fig, "intro_city_controls")
    record("intro_city_controls", source, data, data,
           provenance="User-reported server 5-fold city-CV summary; no local refit",
           statistic="Reported aggregated means; aggregation retained from supplied summary",
           baseline_definitions={"shape": "random-neighbour response similarity",
                                 "amplitude": "refit training-label mean",
                                 "direction": "refit training-label majority sign"},
           intervals="Not supplied; no interval inferred", pairing="visual comparison of summaries",
           claims_excluded=["formal R-squared", "confidence intervals", "neural city-CV advantage"])

def other_figures():
    data=pd.read_parquet(ROOT / "data/active/causal/treatment_unit_list.parquet",columns=["opening_month"])
    counts=data.opening_month.astype(str).str[:4].astype(int).value_counts().sort_index()
    assert counts.sum() == len(data)
    fig, ax=plt.subplots(figsize=(11,3.9),layout="constrained")
    ax.bar(counts.index,counts.values,color="#234E70",width=.72)
    for x,y in counts.items(): ax.text(x,y+8,str(y),ha="center",fontsize=11)
    ax.set_xticks(counts.index, [str(year) for year in counts.index], rotation=35)
    ax.set(xlabel="开通年份",ylabel="站点开通事件数",ylim=(0,counts.max()*1.14))
    ax.grid(axis="y",alpha=.15); ax.set_axisbelow(True); save(fig,"intro_sample_years")
    record("intro_sample_years","treatment_unit_list.parquet",data,data,grouped_counts={str(k):int(v) for k,v in counts.items()})
    fig, axes=plt.subplots(1,2,figsize=(11,3.8),layout="constrained")
    d=np.linspace(0,1,101); p=np.linspace(.05,2.5,101)
    for gamma,style in [(1,"-"),(.5,"--")]: axes[0].plot(d,np.exp(-gamma*d),style,label=f"γ = {gamma}",color="#234E70",linewidth=2)
    for alpha,style in [(0,"--"),(.5,"-")]: axes[1].plot(p,p**alpha,style,label=f"α = {alpha}",color="#168A85",linewidth=2)
    axes[0].set(xlabel="设计不可靠度 D（序数分数）",ylabel="设计分量 exp(−γD)")
    axes[1].set(xlabel="归一化精度 p",ylabel="精度分量 p^α")
    for ax in axes: ax.legend(frameon=False); ax.grid(alpha=.15)
    save(fig,"intro_weight_curve")
    AUDIT["intro_weight_curve"]={"type":"functional illustration","gamma":[1,.5],"alpha":[0,.5],"before_final_normalization":True}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    diagrams(); availability_figures(); pretrend_figures(); event_figures(); history_figure(); city_controls_figure(); other_figures()
    (OUT / "intro_chart_audit.json").write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Exported {len(AUDIT)} figures in PDF / SVG / 300 dpi PNG to {OUT}")

if __name__ == "__main__":
    main()
