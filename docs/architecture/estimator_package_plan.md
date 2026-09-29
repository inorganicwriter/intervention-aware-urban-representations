# 估计器开源包计划（额外 contribution）

状态：已定方案，待训练与消融完成后执行。

本计划把项目内的 Python/GPU 反事实估计器整理为一个独立的开源包，作为论文的
额外 contribution。它不改变论文的因果设计，也不触碰当前被资格指纹绑定的
生产代码。

## 定位

- 形式：独立 repo / PyPI 包（工作名 `urban-causal`），Apache-2.0。
- 卖点：与 R 官方实现做过数值资格认证的 Python/GPU 估计器，附带可复现的
  parity 测试与合成基准。
- 论文表述：open-source, R-parity-certified Python implementation of the
  matching, generalized synthetic control, matrix-completion and event-study
  estimators used in this study。

## 范围

四个估计器全包，另加最小面板契约与推断工具：

| 新包模块 | 来源（项目内） | 内容 |
|---|---|---|
| `matching` | `causal/gpu/matching.py`、`causal/gpu/abadie_imbens.py` | 冻结控制设计（Mahalanobis、M=5 + 静态精炼）、平衡与 holdout/placebo 门禁、Abadie–Imbens 偏差修正 ATT |
| `gsc` | `causal/gpu/gsc.py` | 交互固定效应、rolling CV（1SE）、参数 bootstrap、诊断 |
| `mc` | `causal/gpu/matrix_completion.py` | 核范数目标、lambda CV、unit jackknife（fect 等价） |
| `event_study` | `causal/event_study.py`、`causal/pooled_event_study.py` | 吸收式 TWFE、CRV1（grid/city）、联合 Wald、规范窗口、pooled 路径聚合 |
| `panel`、`inference` | `causal/gpu/contracts.py`、`causal/gpu/inference.py` | 最小 PanelData 契约、CRV1/jackknife/bootstrap 工具 |

## 留在项目内（不进开源包）

队列路由与调度、Response Artifact 契约、空间 donor 排除与城市几何、资格凭证
与 waiver、数据源路径。开源包只接受标准面板输入并返回估计量、标准误、区间与
诊断，不涉及项目的数据治理。

## 硬约束

- **不原地重构** `gpu/` 与 `event_study.py`：它们被资格指纹/hash 绑定，改动会
  作废生产资格。抽取方式为复制到新包后清理耦合，项目内代码保持冻结。
- 新包单测从现有 `tests/unit/` 对应用例迁移并补充合成数据示例。
- parity 测试需要本机安装 R 包（`PanelMatch`、`Matching`、`gsynth`、`fect`、
  `fixest`）才运行，默认 skip；不打包也不分发任何 R 代码。

## 里程碑（顺序）

1. 新建 repo 骨架：pyproject、Apache-2.0、ruff/mypy/pytest CI、CPU 回退。
2. 迁移 `panel` + `inference` + `matching`（含 AB ATT）与单测。
3. 迁移 `gsc`、`mc` 与单测。
4. 迁移 `event_study`（TWFE + pooled）与单测。
5. parity harness：合成面板上与 R 官方实现对拍，产出基准表。
6. 文档：方法公式（适配 `docs/research/econometric_methods.md` 与
   `estimator_formulas.md`）、API 参考、示例 notebook。
7. 发布草稿：README、CHANGELOG、PyPI 预发布；论文引用与 Data/Code
   availability 段落同步更新。

## 触发条件

训练主版本与 W0 / W-full 消融完成、结果确认后启动第 1 步。
