# 稳健性检验

统一入口：`scripts/causal_r/run_robustness_checks.py`

输出目录：`outputs/robustness/`

## 0. 推断有效性与聚类

同一城市的网格共享土地政策、经济增长和规划预期等城市冲击。事件研究同时报告：

- 网格聚类：`cluster ~ grid_id`；
- 城市聚类：`cluster ~ city_key`。

GSC 和 MC 聚合同时输出网格级与城市级预趋势联合检验，城市级结果写入
`joint_pretrend_tests_city`。

## 1. 空间排除半径

主规格排除处理站点网格多边形 1km 内的 donor。敏感性规格使用 1.5km 和
2km。

```bash
python scripts/causal_r/run_robustness_checks.py --spatial-exclusion
```

输出写入 `outputs/robustness/spatial_exclusion/exclusion_{radius}m/`。正式报告比较
不同半径下的 donor 支持、样本构成和 ATT 路径。

## 2. Anticipation 窗口

主规格为 6 个月，敏感性规格为 0 个月和 12 个月。

```bash
python scripts/causal_python/run_causal_label_queue.py \
  --start-order 1 --max-tasks 1 --anticipation-months 0
python scripts/causal_python/run_causal_label_queue.py \
  --start-order 1 --max-tasks 1 --anticipation-months 12
```

正式报告比较 0、6、12 个月窗口的可用样本、估计路径和动态响应。

## 3. 处理前窗口长度

月度主规格使用 36 个月处理前窗口，敏感性规格使用 24 个月和 48 个月。

```bash
python scripts/causal_r/run_robustness_checks.py --window-length
```

该检查验证 24、36、48 个月事件日历。正式比较必须在控制设计与标签规格中使用
相同的窗口参数。

## 4. 协变量集

主规格包含处理前结果历史、区位和轨道特征。敏感性规格分别移除轨道特征和区位
特征。

```bash
python scripts/causal_r/run_robustness_checks.py --covariate-set
```

比较内容包括控制选择、结果历史平衡、静态协变量平衡、holdout 误差和 placebo
门禁结果。

## 5. Donor 范围

主规格采用同城优先的六轮路由。跨城规格使用仅由处理前数据拟合的标准化参数，
并与同城结果分开报告。

```bash
python scripts/causal_r/run_robustness_checks.py --donor-pool
```

比较内容包括 donor 支持、匹配接受率、方法路由、平衡诊断和动态响应。

## 6. 竞争事件

主处理定义为网格内首个正式地铁站开通。竞争事件敏感性剔除存在其他轨道事件
标记的网格后重新估计。

```bash
python scripts/causal_r/run_robustness_checks.py --competing-events
```

正式报告并列给出全样本和剔除竞争事件后的结果。

## 7. Spillover 与网络规模异质性

事件研究按同期开通站数 `stations_opened_same_month` 的中位数分层，比较小规模
与大规模开通批次。输出写入：

```text
outputs/event_study/matching/{family}/spillover_{stratum}_*.csv
```

该分层用于描述网络规模异质性，空间排除半径敏感性单独报告。

## 正式执行要求

- 所有规格使用同一冻结处理清单和结果定义；
- 每个规格单独保存配置、输入哈希、样本构成和方法路由；
- Matching、GSC 和 MC 分开汇总，不合并标准误；
- smoke 输出用于验证执行合同，论文估计使用正式结果；
- 正式比较在 Response Artifact 发布后运行，并报告不可用任务及原因。

## 文献

1. Abadie, A., Athey, S., Imbens, G.W. & Wooldridge, J.M. (2023). When should you adjust standard errors for clustering? *Quarterly Journal of Economics* 138(1): 1-35.
2. de Chaisemartin, C. & D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *American Economic Review* 110(9): 2964-2996.
3. Yu, N., de Jong, M., Storm, S. & Mi, J. (2013). Spatial spillover effects of transport infrastructure: evidence from Chinese regions. *Journal of Transport Geography* 29: 56-66.
