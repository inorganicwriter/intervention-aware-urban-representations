# S/T/C 真实数据服务器验收回执

日期：2026-10-19 UTC
结论：**正式服务器工程验收完成**。本次短训练用于工程闭环，不据此宣称模型性能或泛化能力。

当前有效结果见文末“最新验收：114650Z”。本文前面的较早运行均为历史记录，不能替代最新结果。

## 范围与指纹

- 主机：222.29.102.59:2222；项目：/home/nas/lsr/MIT_summer
- 运行目录：/home/nas/lsr/MIT_summer/outputs/stc_server_acceptance/stc_acceptance_20261019T050410Z
- 使用既有 mit 环境，未重建环境或全量升级。最小安装 rasterio 1.4.4、umap-learn 0.5.12、scikit-learn 1.9.1 及缺失传递依赖；详情见 environment_after_dependencies.txt。
- 严格生产候选代码：candidate/；归档 SHA256 bae40c2c1729d8738eb37feb25bc961a8f5bd99aff7da7c327cb3112be98c532；源码清单 SHA256 701d493b72d948e667b3edcde679c9a4253af7ce6b76fdebd7b91909c7946037；441 文件通过校验，0 失败。
- 正式 Response Artifact SHA256 aa7df0028cc42c3acb33d106a8cf0841602660864da8d34253087b29d2a8464；发布清单 SHA256 4f247b8dc80ba8a2ecb7a4c214d65fa4a615e553f67ac44d350bdb86ca67e635。两类报告均记录 strict_production=true、is_synthetic=false，并确认模块来自隔离候选目录。

## VIIRS 与 POI 闭环

两类各选取64个真实事件，覆盖多城市并保留正式城市划分；事件身份、城市和 split 见 selected_events_by_family.json。

街景：31张图全部读成功，来自16个有图事件；48个事件为合法缺失；读取失败和来源不一致均为0。卫星序列固定为开通前-6至-1年：383/383张可用TIFF实际读成功，另有1个合法缺年帧；读取或时间错误为0。完整64维 AlphaEarth 被背景分支消费。

| 结果族 | 维度 | 有效监督单元 | 各维支持 |
|---|---:|---:|---|
| VIIRS | 6 | 381 | t1=64, t3=64, t6=64, t12=61, t18=64, t24=64 |
| POI | 12 | 720 | 4个指标均为 t1=64, t2=60, t3=56 |

前向/反向、有限损失和梯度、有效分支梯度、固定S改变T/C时z不变、视觉分支确实被消费、mask扰动不变均通过；mask输出最大差为0。每类完成2轮训练、检查点保存和完整契约恢复，并导出64个唯一事件。恢复z、预测及导出z最大误差均为0；导出z维度64、融合表示维度160。

当前损失对照正式Response Artifact中的开通后时间窗 causal_response_label、mask和权重。开通后街景独立归档，不进入开通前状态，也不是当前损失目标。随后已单独构造事件、固定网格、目标跨度和模态的视觉转移配对索引；索引不会改变上述因果响应监督。

## 回归与资产保护

指定六个测试文件全部通过：60 passed、0 failed、0 skipped、0 deselected、0 excluded，127.68秒。UMAP真实测试实际执行并断言 method == umap；PCA回退有独立测试。

验收前后32个正式输入、响应发布、街景/卫星索引及生产队列文件的SHA256与字节数完全一致。

## 证据路径

- 机器摘要：reports/acceptance_summary.json
- 运行命令（不含凭证）：reports/executed_commands.txt
- 回归日志：logs/regression_tests.log
- VIIRS报告/检查点/导出：results/viirs/smoke_verification_report.json、results/viirs/training/best_model.pt、results/viirs/embeddings.parquet
- POI报告/检查点/导出：results/poi/smoke_verification_report.json、results/poi/training/best_model.pt、results/poi/embeddings.parquet
- 保护资产指纹：reports/baseline_production_asset_hashes.json、reports/after_production_asset_hashes.json

## 修复记录

迭代期间保留了真实错误证据：AlphaEarth正式列名适配（v4）、响应样本与官方sample_index划分合并（v5）、mask扰动导致街景图像路由变化（v6）。v7修复后按未放宽的严格门禁重跑VIIRS，并完成POI与全部回归测试。

## 独立视觉状态转移配对索引（2026-10-19）

代码位于项目 `scripts/analysis/build_visual_transition_pair_index.py`，SHA256 `c514602655fbcd2859a7a5f13fae4866edc645aab5e9145f4e47def6e1734318`。代码直接放在项目脚本目录；服务器 `/tmp` 中没有副本。

本次索引路径为 `/home/nas/lsr/MIT_summer/outputs/stc_server_acceptance/stc_acceptance_20261019T065810Z/visual_transition_pairing/`，覆盖全部 5,048 个处理事件，保留 0..3 四个目标跨度与街景、卫星两种模态，共 40,384 行。mask 计数如下：

| 模态 | event_time | 前状态可用 | 后目标可用 | 有效监督配对 |
|---|---:|---:|---:|---:|
| 卫星 | 0 | 5,048 | 0 | 0 |
| 卫星 | +1 | 5,048 | 4,738 | 4,738 |
| 卫星 | +2 | 5,048 | 4,351 | 4,351 |
| 卫星 | +3 | 5,048 | 4,045 | 4,045 |
| 街景 | 0 | 3,778 | 504 | 433 |
| 街景 | +1 | 3,778 | 1,744 | 1,526 |
| 街景 | +2 | 3,778 | 1,688 | 1,391 |
| 街景 | +3 | 3,778 | 1,553 | 1,077 |

`event_time=0` 卫星目标全部 mask 掉，以排除可能混合开通前后的年度合成；街景按真实 `YYYY-MM` 月份要求目标拍摄月晚于开通月。每个后状态均为 target-only，`post_target_predictor_use_allowed=false`。Parquet SHA256：`d6184ccb65425aa486575270db8f486b610cf0af4afc4e149bf6dcd58e78a3e3`；审计 JSON SHA256：`18bc98871ea6e9c302e869e3a7c57a809301c428d1b10addbcb45ce0ba46bb9a`。详细来源哈希、时间字段与 mask 定义见项目 [`visual_state_transition_pair_contract_2026-10-19.md`](../research/visual_state_transition_pair_contract_2026-10-19.md)。本项仅建立配对数据和监督 mask，没有启动视觉状态转移训练，也没有修改 causal_response 标签、现有 predictor 或训练损失。

## 最新复核：真实模态闭环与街景多图输入

以下复核运行晚于上文记录，作为当前状态：

- 主机：`222.29.102.59:2222`；项目：`/home/nas/lsr/MIT_summer`；运行目录：`/home/nas/lsr/MIT_summer/outputs/stc_server_acceptance/stc_acceptance_20261019T091255Z`。
- 使用现有 `mit` 环境（Python 3.13、PyTorch 2.11.0+cu130，CUDA 可用，4 张 RTX 4090）；本轮未创建环境或改动依赖。
- `conda run` 启动器重复提示 base 环境中的 `RequestsDependencyWarning`（urllib3/chardet 与 requests 支持范围不匹配）；mit 内真实读取、训练、导出和回归均成功。本轮没有升级依赖。
- 隔离候选源码包 SHA256：`8d1a6d62e31df9cd1ad06d4bedc2f41066a54a608fda04f007d74ab58d35b3a8`。关键源码逐文件 SHA256 在 `reports/critical_source_sha256.txt`；候选模块导入路径均指向本轮 `candidate/src`。
- 正式生产 Response Artifact SHA256：`aa7df0028cc42c3acb33d106a8cf0841602660864da8d34253087b29d2a84642`；发布 manifest SHA256：`4f247b8dc80ba8a2ecb7a4c214d65fa4a615e553f67ac44d350bdb86ca67e635`。两类报告均标为非合成真实数据并使用严格生产发布。

| 结果族 | 事件数 | 响应维度 | 有效监督单元 | 各维支持 |
|---|---:|---:|---:|---|
| VIIRS | 64 | 6 | 381 | t1=64、t3=64、t6=64、t12=61、t18=64、t24=64 |
| POI | 64 | 12 | 720 | 4 个结果 × t1=64、t2=60、t3=56 |

至少有一个有效监督单元的事件按正式城市划分计数：VIIRS train=46、validation=9、test=9；POI train=44、validation=11、test=9。各城市×划分的逐项有效事件数，以及每个维度按划分统计的数量，见对应 smoke 报告的 `tests.response_supervision.valid_events_by_city_and_split` 与 `per_split_counts`。

两类运行均实际读取 383 张卫星 TIFF，并保留 1 个合法缺年；无读取失败。街景实际读取 31 张图片，覆盖 16 个事件；48 个无图事件按合法缺失处理，读取失败和来源不一致为 0。完整 64 维 AlphaEarth 进入背景分支。样本身份、城市和官方划分见各族 `results/{viirs,poi}/data/sample_index.parquet`；逐城市和划分支持数量保存在对应 smoke JSON 报告中。

VIIRS 与 POI 各完成两轮 CUDA 短训练、保存恢复和 64 个唯一事件导出；恢复预测、状态向量及导出向量最大差均为 0。VIIRS 检查点 SHA256 为 `e58f0664e1d27d208e4e331476b650ff6b2d17dd57c11cc7d9989bc022e30a49`，导出 SHA256 为 `534a97bf6633c0b289f3b09bb0288ffd181b38f8229e33647882ad4958b3a0d0d`。POI 检查点 SHA256 为 `55e45aa6d105ca9731ee917b538dc449f84ac49b211301fe1e3630e9540f0e2b3`，导出 SHA256 为 `4789c5547215fee62392f76311c4bca1612c6b60796cf0672009028155cd1aacc`。完整报告分别位于 `results/viirs/smoke_verification_report.json` 和 `results/poi/smoke_verification_report.json`。

指定六个回归文件共 60 项：60 passed、0 failed、0 skipped、0 deselected、0 excluded，耗时 189 秒；测试日志中有 24 项依赖或数值库警告，不影响通过结论。日志为 `logs/regression_tests.log`；不含凭证的主命令为 `reports/executed_commands.txt`。验收前后 18 个受保护输入、索引与队列文件指纹一致，`reports/protected_assets_diff.txt` 为空。

本轮重建配对索引覆盖 5,048 个事件、4 个跨度和 2 个模态，共 40,384 行。开通前街景输入探针对 6 个有效配对实际加载 12 张源图，另加载 12 张开通后 target-only 目标图；时间压缩值为 0.333–0.600，时间感知集合编码产生有限的 6×64 表示。预测编码器只收到开通前图片、逐图时间差和 mask，开通后图片没有输入预测编码器。配对 Parquet SHA256 为 `a15a9b73802f627981d0110a53ca3dd37b405c2da4464ad51443b9bb2ad6dc65`；探针记录为 `reports/streetview_transition_input_probe.json`。

本轮街景实现新增代码没有引入项目自定义版本号标记。开通后的视觉目标仍只供单独的视觉状态转移监督；转移头和训练循环尚未实现，当前 S/T/C 模型损失仍由真实因果响应标签监督。

## 最新验收：114650Z

**结论：正式服务器工程验收完成。** 最新运行目录：`/home/nas/lsr/MIT_summer/outputs/stc_server_acceptance/stc_acceptance_20261019T114650Z`。使用服务器既有 `mit` 环境（Python 3.11、PyTorch 2.11.0+cu130、CUDA、4×RTX 4090），本轮未修改环境依赖。

### 表征目标与数据

按《研究思路.docx》的响应感知表示学习目标，开通前多模态状态 `S` 编码成一个定长响应敏感向量 `z=f_theta(S)`；正式准实验因果响应标签监督 `g_phi(z,T,C)` 的响应预测，并用于响应相似性对齐。`T` 与 `C` 是响应预测条件。当前原始 `C` 为82个数值字段（AlphaEarth64 + 18个表格背景字段）和19个 mask；编码后得到64维 `b`，与64维`z`、32维`t`拼接为160维响应预测输入。此目标不训练视觉开通前后转移，也不把开通后街景或卫星图作为预测目标。街景集合按距开通的日历月数 `d` 计算时间差，以 `d/(d+12)` 压缩并由时间感知注意力聚合。

正式 Response Artifact SHA256：`aa7df0028cc42c3acb33d106a8cf0841602660864da8d34253087b29d2a84642`；manifest SHA256：`4f247b8dc80ba8a2ecb7a4c214d65fa4a615e553f67ac44d350bdb86ca67e635`。VIIRS、POI共用此 strict-production 发布，均为真实标签。

| 结果族 | 事件/城市 | 结果维度 | 有效监督 | train / validation / test 有效事件 |
|---|---:|---:|---:|---:|
| VIIRS | 64 / 40 | 6 | 381 | 46 / 9 / 9 |
| POI | 64 / 37 | 12 | 720 | 44 / 11 / 9 |

VIIRS各维有效数为`t1=64, t3=64, t6=64, t12=61, t18=64, t24=64`；POI四个结果各自为`t1=64, t2=60, t3=56`。每个族均实际读取31张街景（16个事件），48个事件为合法街景缺失，读取失败为0；卫星`-6..-1`年序列实际读取383/383张可用TIFF，另1个合法缺年，读取/时间失败为0。街景开通间隔1–30个月；VIIRS间隔均值10.387个月，POI 9.097个月。完整AlphaEarth64维背景分支通过消费检查。

### 训练、恢复、导出与回归

| 结果族 | 训练 | 检查点 SHA256 | 导出 SHA256 | 导出事件 |
|---|---:|---|---|---:|
| VIIRS | 2 epochs, CUDA | `44725bffe66e51ddd760f7e89b8848f2a0e2865b21b4c958c7f45faa86656274` | `212289e0c178aa6fec02827bb69f21b70519d9e0fd8a98bf4ff60c1a9fe2dfae` | 64 unique |
| POI | 2 epochs, CUDA | `d4ff16a92f798a154535eba131c5db9ee403b1a58be5287225b30617985091a5` | `f63075a167efd6808e308df595e3322097e8500f9f24f9db3519f5c1ef91c74a` | 64 unique |

两类检查点均验证完整输入契约；恢复后的`z`、响应预测与导出向量最大误差均为0。`z`维度64，融合预测输入维度160；固定`S`改变`T/C`时`z`不变；mask输入扰动不影响结果；街景、卫星和AlphaEarth均通过分支消费检查。指定六个测试文件共`62 passed, 0 failed, 0 skipped, 0 deselected, 0 excluded`；真实UMAP测试执行，PCA回退由独立测试覆盖。验收前后18个受保护输入、发布、索引和生产队列文件的SHA256一致，差异文件为空。

首次回归尝试的错误证据保存在`logs/regression_tests_initial_failures.log`：服务器起初没有DINOv2缓存，测试触发联网获取权重超时；另有冻结的测试源码哈希未随已审阅训练器更新。随后把本地已有的官方DINOv2缓存复制到本次运行目录并校验178个文件，更新过期的冻结哈希，针对性图像敏感性测试通过，最后六文件全量回归通过。最新运行没有修改`mit`环境依赖；更早的首轮服务器尝试曾在同一环境最小安装`rasterio`、`umap-learn`和`scikit-learn`，记录见首轮历史回执。

候选源码447个文件校验通过，tree SHA256：`8983f16c51edb6ee804b5896f5255d2fbb04711173613d3a12651caf9a5dd844`；候选归档 SHA256：`da75698172baf324d2821c40784bc30071ece4a5e1d123f4adaddc3b8bb54113`。DINOv2项目内缓存178个文件校验通过，权重 SHA256：`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46e7a1d8d00217f2929aef60cd9`。

### 证据路径

- 机器验收摘要：`/home/nas/lsr/MIT_summer/outputs/stc_server_acceptance/stc_acceptance_20261019T114650Z/reports/acceptance_summary.json`
- 无凭证命令：`.../reports/executed_commands.txt`
- 回归日志：`.../logs/regression_tests_final.log`
- VIIRS报告、检查点、导出：`.../results/viirs/smoke_verification_report.json`、`.../results/viirs/training/best_model.pt`、`.../results/viirs/embeddings.parquet`
- POI报告、检查点、导出：`.../results/poi/smoke_verification_report.json`、`.../results/poi/training/best_model.pt`、`.../results/poi/embeddings.parquet`
- 逐图时间差统计：`.../reports/streetview_temporal_gap_summary.json`
- 代码及资产哈希、资产保护比较：`.../reports/critical_source_sha256.txt`、`.../reports/baseline_production_asset_hashes.sha256`、`.../reports/after_production_asset_hashes.sha256`、`.../reports/protected_assets_diff.txt`

以上短训练只证明真实数据读取、因果监督、训练、保存恢复和导出工程闭环可运行，不作为模型性能或跨城泛化结论。
