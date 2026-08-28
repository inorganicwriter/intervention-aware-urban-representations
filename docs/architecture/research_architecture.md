# 项目核心研究架构

状态：有效  
更新日期：2026-08-27

## 1. 研究目标

项目研究不同地点对同类城市干预的响应相似性。
当前干预为地铁站正式开通，最终任务是学习可跨城市迁移的
intervention-conditioned urban representation。

## 2. 两层研究系统

项目分为两层，分别管理：

1. **反事实与标签层**：为每个处理网格寻找或构造可信的未处理反事实，生成带质量信息的局部响应标签；
2. **表征学习层**：使用处理前多模态城市特征、干预信息和响应标签训练可迁移表示。

第一层的输出是第二层的监督信号。总体平均处理效应、可行性回归和旧 DID 结果归入补充分析，
逐网格响应标签作为表示学习的训练目标。

## 3. 核心数据流

```text
原始数据与来源记录
        ↓
标准化网格、站点事件和多模态协变量
        ↓
固定 5,048 个处理网格与空间合格 donor universe
        ↓
只使用处理前信息的控制设计
        ↓
单一匹配控制
        ↓ 质量检查未通过
Xu generalized synthetic control
        ↓
观测轨迹 - 反事实轨迹
        ↓
Response Artifact（标签、质量、缺失掩码、不确定性）
        ↓
干预条件城市表征学习
```

## 4. 模块边界

- `data/archive/raw/`：不可变原始材料；
- `data/archive/staging/`：解析后、尚未发布的数据；
- `data/active/curated/`：标准化协变量、站点和空间产品；
- `data/active/causal/`：固定处理清单、设计输入和生产队列；
- `data/active/labels/`：观测结果变量；
- `outputs/`：审计、模型对象和响应标签结果；
- `src/urban_intervention/`：可复用 Python 实现；
- `scripts/causal_r/`：正式计量估计器和队列入口；
- `tests/`：数据合同、信息边界和估计器门禁。

## 5. 固定研究约束

- 处理单元固定为 500m × 500m 网格；
- 处理事件为网格内正式地铁站点开通；
- 主空间污染排除半径为 1km；
- 控制选择只读取处理前结果、处理前缺失状态和冻结空间规则；
- 匹配失败后进入 GSC，GSC 失败后进入 MC，三者均失败才明确跳过；
- 房价、VIIRS、POI、人口允许使用不同的反事实和缺失掩码；
- 任何标签必须能追溯到处理单元、控制设计、输入版本和质量诊断；
- 试运行结果只有在通过正式检查并标记为可发布后，才能进入训练标签和研究结论。

## 6. 权威文件

- 项目总览与文档权威层级：`docs/PROJECT_GUIDE.md`；
- 因果设计与响应标签：`docs/research/counterfactual_response_label_design.md`；
- DID/事件研究诊断：`docs/research/identification_and_diagnostics.md`；
- 冻结研究决策：`docs/research/decisions/`；
- 数据合同：`docs/data/`；
- 当前运行入口：`scripts/causal_r/README.md`；
- 动态运行状态：`docs/operations/current_project_status.md`。

具体算法参数只在因果主设计和 DDR 中维护，本文件不重复这些细节。
