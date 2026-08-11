# 安居客小区历史价格采集规划

> 目标：零购买路径下补齐 2012–2018 小区级房价数据（重点 16 个缺口城市）。
> 更新：2026-08-10

## 0. 背景与目标

现有房价缺口根因：购买的链家成交只覆盖 7 省市，44 城中 16 城（长沙、南昌、贵阳、
南宁、兰州、呼和浩特、乌鲁木齐、太原、哈尔滨、昆明、福州、石家庄、长春、西安、
沈阳、大连）2018 年前月度观测 < 100。wayback 已挖尽（archive.org 对 2018 年归档
趋零）；creprice/安居客均被 58 系 antibot 拦截；学术平台无合适数据集。

安居客小区详情页公开提供**小区级历史月度挂牌均价**（多数小区 2012 年起），
这是当前唯一能实质性补 2012–2018 小区级数据的零购买路径。

**可验收目标（阶段 3 结束时）：**

| 指标 | 目标值 |
|---|---|
| 采集到历史价格的小区数 | ≥ 100,000 |
| 覆盖 2012–2018 区间的小区数 | ≥ 60,000 |
| 16 个缺口城市中每城有 2012–2018 数据的小区数 | ≥ 500 |
| 新增月度观测（小区-月） | ≥ 500 万 |

## 1. 现有基础设施盘点（已就绪）

- **小区注册表**：166,079 小区（44 城全覆盖，每城 1,373–9,245），字段含
  `normalized_name`、`aliases`、`district`、`centroid_lon/lat`、
  `has_anjuke_boundary`（58.3%）、`anjuke_source_id`（内部 MD5，非页面 ID）
- **网格桥接**：`community_grid_bridge.parquet`，160,278 小区（96.5%）已桥接 500m 网格
- **采集基础设施**：playwright + playwright-stealth 已装；`.runtime/browser_profiles`
  存在；`wayback_research_scraper.py` 的 inventory/manifest/断点续爬模式可复用
- **解析/入库管线**：`build_anjuke_label.py`（截面版）、`build_wayback_label.py`、
  `build_housing_price_panel.py` 已存在，新数据走同一 admission 流程

## 2. 需要用户准备的前置资源（进入阶段 0 前）

| 资源 | 说明 | 预估费用 |
|---|---|---|
| 住宅代理池 | ≥ 50 IP（按量套餐更优），覆盖多运营商 | 500–1500 元 |
| 打码平台 | 图鉴/超级鹰/云码等，支持点选/滑块验证码 | 300–800 元 |
| 干净出口 | 当前 IP（94.119.32.6）已被 58 拉黑，需 VPN/新网络 | 0–100 元 |
| **预算上限确认** | 阶段 0 止损线 300 元；阶段 1 总预算需批准 | — |

## 3. 阶段划分与验收门槛（Go/No-Go）

### 阶段 0：试点验证（1–2 天，预算 < 300 元）

| 子任务 | 内容 | 产出 |
|---|---|---|
| 0a. ID 发现 | 验证小区列表页 `https://{city}.anjuke.com/community/p{n}/` 或搜索接口的 ID 格式（数字/hex）、分页结构与总量 | 南昌小区 ID 全集（含区县、名称） |
| 0b. 反爬验证 | 1 城 10–20 个小区详情页：验证码触发率、打码通过率、代理轮换效果 | 反爬参数表 |
| 0c. 数据格式 | 确认历史价格数据位置（页面内嵌 JSON / AJAX 接口）、时间粒度（月度）、2012–2018 深度 | 解析器原型 |
| 0d. 名称匹配 | 抓取小区名 vs `registry.normalized_name`/`aliases` 匹配率 | 匹配策略定稿 |
| 0e. 单城试跑 | 南昌全量（~2,000 小区）跑通：采集→解析→匹配→网格聚合 | 端到端演示数据 |

**Go/No-Go 门槛（全部满足才进入阶段 1）：**

- [ ] 打码 + 代理下详情页成功率 ≥ 80%
- [ ] ≥ 60% 试点小区有 ≥ 36 个月历史价格
- [ ] ≥ 30% 试点小区覆盖 2018 年之前
- [ ] 名称匹配率 ≥ 85%（名称 + 区县 + 城市三键）
- [ ] 单城端到端管线跑通

任一不达标 → 止损（投入 < 300 元），转方案 B（接受缺口，GSC/MC 兜底）。

### 阶段 1：全量采集（3–7 天）

- 1a. 列表页采集：44 城全量小区 ID（预估 8,000–12,000 页）
- 1b. 详情页采集：约 14–16.6 万小区详情页（与注册表小区名交集优先）
- 1c. 存储：raw HTML（zstd 压缩）+ inventory JSON（复用 wayback 的
  exact-capture 模式，含 timestamp/original_url 溯源）
- 1d. 监控：成功率、封禁率、打码消耗实时统计；**第 2 天检查点**：
  完成率 > 30% 且封禁率 < 20% 才继续，否则调整策略或止损

### 阶段 2：解析与入库（1–2 天）

- 2a. 解析历史价格序列（小区-月-均价 + 元/㎡）
- 2b. 名称匹配 registry（normalized_name + aliases + district + city_key）
- 2c. 生成 `data/active/labels/housing/listing_price/anjuke_history/{city}/`
- 2d. 质量审计：异常值（跳变 >50%）、断点率、覆盖率报告，走人工质量决议

### 阶段 3：面板集成与验证（1–2 天）

- 3a. 网格聚合（复用 `community_grid_bridge` 权重）
- 3b. 交叉验证：与链家成交/wayback 同小区同期价格比对（Pearson r、中位差）
- 3c. 16 缺口城市 2012–2018 覆盖改善评估（对照阶段 0 基线）
- 3d. 决定是否纳入正式房价标签族（admission 决议 + DDR 记录）

## 4. 反爬策略

| 项 | 方案 |
|---|---|
| 请求速率 | 每 IP 3–5 s/请求；并发 8–16（随代理池规模） |
| IP 轮换 | 每 20–50 请求轮换；被封 IP 冷却 30 分钟 |
| 验证码 | 检测 → 打码平台 → 回填重试（最多 2 次）；滑块/点选由平台能力决定 |
| 指纹 | playwright stealth + UA/视图/时区/语言随机化 |
| 合规 | 仅公开页面；不登录、不采集个人信息（仅小区级聚合价格）；低频 |

## 5. 成本估算

| 项 | 估算 | 说明 |
|---|---|---|
| 住宅代理 | 500–1500 元 | 17 万请求，按量计费 |
| 打码 | 300–800 元 | 验证码触发率按 10–30%，单次 0.01–0.03 元 |
| 人工 | 5–12 天 | 含监控与故障处理 |
| **合计** | **约 1000–2500 元 + 5–12 天** | 阶段 0 先花 <300 元验证 |

## 6. 风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| antibot 升级（滑块/行为检测） | 中 | 打码平台扩展；降低频率拉长周期 |
| 部分小区只有近 12 月价格 | 高 | 接受；标签质量分级（只标"近 12 月"与"历史完整"） |
| 大规模封 IP | 中 | 代理池扩容、冷却、域名限速 |
| 名称匹配率低 | 中 | aliases + 区县 + 坐标（centroid 距离 <1km）辅助 |
| 数据质量差（价格跳变/断档） | 中 | 质量审计 + 人工决议；不合格小区不入面板 |

## 8. 代码实现状态（2026-08-10）

采集代码已实现并通过单元测试（构造数据验证解析/匹配逻辑），位于：

```
scripts/collection/anjuke_history/
  config.py          配置：代理/打码/速率（全部环境变量注入，仓库无密钥）
  collector.py       列表页 ID 发现 + 详情页采集（断点续爬、代理轮换、
                     封禁冷却、验证码钩子、进度监控）
  parser.py          历史价格解析（nested_series / json_series / regex_pairs
                     三策略，试点后锁定真实选择器）
  matcher.py         registry 匹配（exact / alias / loose 三级）
  build_labels.py    解析→匹配→质量门→labels 发布
scripts/collection/run_anjuke_history.py   CLI（discover/collect/parse/build/all）
```

运行方式（资源就绪后）：

```powershell
$env:ANJUKE_PROXY_FILE = "D:\path\proxies.txt"     # 每行一个住宅代理
$env:ANJUKE_CAPTCHA_API_KEY = "<打码平台key>"       # 留空=无打码模式
$env:ANJUKE_REQUEST_INTERVAL = "3"

# 阶段 0 试点（南昌 1 城，先 --limit 验证）
conda run -n mit python scripts/collection/run_anjuke_history.py --stage discover --city nanchang --limit-pages 5
conda run -n mit python scripts/collection/run_anjuke_history.py --stage collect --city nanchang --limit-ids 20
conda run -n mit python scripts/collection/run_anjuke_history.py --stage build --city nanchang

# 全量
conda run -n mit python scripts/collection/run_anjuke_history.py --stage all --city all --workers 8
```

输出：
- `data/archive/raw/housing/anjuke_history/` — 原始 HTML + ID 清单 + JSONL manifest（可断点续爬）
- `data/archive/staging/anjuke_history/matched/` — 匹配审计
- `data/active/labels/housing/listing_price/anjuke_history/{city}/` — 正式标签

**待办（试点后）**：按真实页面结构锁定解析器选择器；按实际验证码类型接线打码平台后端（`solve_captcha` 预留接口）；按试点结果调 `LIST_PAGE_SIZE` 与质量门。

## 7. 验收（投入有效性的最终判据）

阶段 3 结束时对照 §0 目标表逐项核验。核心判据：
**16 个缺口城市中 ≥ 14 城进入 2012–2018 有数据状态（每城 ≥ 500 小区），
且新增小区-月观测 ≥ 500 万、交叉验证中位价差 < 15%。**
未达标的城市如实写入论文数据支持边界，不强行合并。

