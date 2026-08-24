# DDR-005：城市中心注册表（学术口径，McMillen 2001 完整方法）

状态：frozen

日期：2026-08-05

## 1. 目的

为训练前数据集提供"到城市中心距离"与"到最近副中心距离"特征所需的冻结中心坐标。
口径为学术定义（McMillen 2001 非参数局部峰值法 + Giuliano–Small 规模门槛），
不使用任何官方/行政中心假设，且只用处理前信息，无泄漏风险。

## 2. 方法

### 2.1 密度面（Step 1）

- 数据：`data/active/curated/poi/{city}_poi_grid_yearly.parquet` 的 `poi_count`，
  冻结窗口 **2012–2015**（处理前，时不变）；
- 变换：`log1p(年均 poi_count)`（POI 有零值，替代文献的 log 就业密度，声明偏差）；
- 估计：**局部二次 LWR**（locally weighted regression），高斯核，逐点**局部中心化**
  设计 `(u-u_i)/h`，投影坐标 u/v（km，cos(lat) 纬度校正）；
- **带宽冻结为 1.5km**：LOO-CV 最优值落在 σ 网格边界（0.75km，欠平滑过拟合），
  属于 500m 网格噪声数据的已知行为，故 CV 结果作为诊断记入 manifest，
  敏感性报告覆盖 0.75–3.0km；
- **逐点均值面方差**（sandwich 形式）：
  `Var(ŷ_i) = s²·e1'(X'WX)⁻¹(X'W²X)(X'WX)⁻¹e1`，
  `s²` 为局部加权残差方差（`Σw·e²/(Σw−6)`）；这是检验"平滑面特征"的正确方差
  （预测方差含观测噪声，会系统性低估峰值显著性）。

### 2.2 显著局部峰值（Step 2）

- 峰值：平滑面在 `peak_radius`（1.5km）窗内的严格局部极大；
- 显著性：`t = (ŷ_peak − mean(ŷ_N)) / sqrt(Var_peak + mean(Var_N))`，
  正态近似双侧 p；
- 多重检验：全城候选峰值 **Benjamini–Hochberg FDR 校正（α=0.05）**；
- **内域约束**：`kernel_mass > 0.5` 的格才参与（消除城界截断窗口的边缘伪峰）。

### 2.3 规模门槛（Step 3，Giuliano–Small）

- 支撑面积：峰值所在**8 连通分量**中 `ŷ ≥ floor`（50th 分位）且位于
  `peak_radius` 窗内的格数 ×0.25km²，要求 ≥ **5 km²**；
- 与主中心距离 ≥ **3km**；数量上限 **8**（按平滑密度排序）。

### 2.4 主中心

全城平滑面全局最大格（内域）。

### 2.5 验证回归（Step 4，Giuliano–Small / McMillen 的验证步骤）

每城：`log1p(D_i) ~ dist_main + dist_sub1..k`（OLS，手工 SE），
输出 `city_centers_validation.csv`（系数/t/p/R²，含主中心-only 基准 ΔR²）。

## 3. 产出

```
data/active/reference/city_centers.csv                 注册表（role, 坐标, 密度, SE, t, p, 支撑面积, 距离）
data/active/reference/city_centers_manifest.json       全部参数 + CV 诊断 + 输入 SHA-256
data/active/reference/city_centers_validation.csv      验证回归
data/active/reference/city_centers_cv.csv              LOO-CV 带宽表
data/active/reference/city_centers_sensitivity.csv     参数变体中心漂移
outputs/figures/centers/{city}_centres.png      密度面 + 中心审计图
```

## 4. 关键结果（2026-08-05，44 城）

- 290 个中心（44 主中心 + 246 副中心），每城主中心地理合理性人工抽查通过
  （北京国贸、上海人民广场、广州越秀、深圳福田、西安钟楼、重庆解放碑等）；
- 敏感性：主中心在带宽变体下 92% 漂移 ≤2km，其余参数 100% ≤2km；
  副中心集合随带宽变化（敏感性表如实报告）；
- 验证回归：北京 r² 0.26→0.40（σ=1.5km，加入副中心后 Δr²=0.14），
  副中心系数显著为负 → 中心有效组织密度场。

## 5. 与文献的偏差（显式声明）

1. POI 总量代理就业密度（数据可得性）；
2. log1p 替代 log（零值）；
3. 局部二次替代 McMillen 的局部线性（更灵活；敏感性含局部线性变体待补）；
4. 带宽冻结 1.5km（CV 诊断性报告，理由见 2.1）；
5. FDR 校正（现代规范，McMillen 原文未做）。

## 6. 下游使用

- 注册表只供训练前特征构造（`build_station_context_features.py`）使用；
- 不进入控制匹配（匹配设计冻结，DDR-003/004 不变）；
- 后续特征：`dist_main_center_km`、`dist_nearest_subcenter_km`（haversine）。
