# STC 共享状态联合训练实施与验收报告

日期：2026-10-22。任务依据：[任务书及A–D补充](stc_joint_training_task.md)、[实施计划](stc_joint_training_plan.md)。

工程改造、完整Python回归及小规模三组对照已完成。服务器strict-production联合包已生成；正式VIIRS-only与POI-only训练和测试评估已完成，Joint训练仍在运行（最近检查为epoch 88/batch 78）。以下单任务指标是当前初步正式测试结果，联合组和支持一致性核验完成后再形成完整比较结论。本地小规模训练数值只用于工程验收，不能作为正式科学证据。

## 已实现合同

- 唯一共享状态表示z128，干预t32，背景b96，预测输入256；独立VIIRS 256→128→6、POI 256→128→12。
- 有效AlphaEarth64直接保留，不降维；缺失使用可学习64维token。availability作为表格编码器的额外二值输入，表格输出32。b不整体L2归一化。z继续L2归一化。
- 两族Anchor分别按有限标签、有限正设计权重归一化；两族有效时各0.5，单族有效时权重1。Alignment按族独立计算共同支持、可靠性和正样本目标，仅平均有效任务。零样本对不是有监督零相似度。
- 全无监督batch在前向之前跳过，既不optimizer.step，也不更新BatchNorm缓冲；有效标签上的非有限预测显式报错；NaN标签、零/非有限权重、缺失SE等均有边界处理和测试。
- SE只进入Alignment的现有可靠性修正；主Anchor不重复叠加逆方差权重。正式数据构建命令显式使用weight-alpha=0。
- 新模式自动识别联合STC发布包，也可显式指定joint/viirs_only/poi_only。三组使用同一18维标签布局、输入表、城市划分、标准化和双头结构，只改变监督任务mask。
- 可配置全局范数裁剪，默认1.0，可关闭；记录裁剪前/后范数、clip scale。分任务梯度以其对实际总loss的贡献计算，包含独立有效任务归一化系数；记录State范数、余弦、冲突。
- 诊断通过autograd.grad观察，不写参数.grad，不额外optimizer.step。测试验证诊断启用/关闭后的实际AdamW更新一致。
- 记录AlphaEarth64/表格32的RMS、条件输入梯度范数、原始输入梯度范数及缺失token/表格编码参数梯度。
- 检查点、配置和导出metadata记录新架构、模式、维度、数据哈希、代码哈希、环境及可用的DINOv2权重SHA256；历史STC检查点仍可加载。
- 导出128个emb列、256个fused列、6+12个预测列、完整事件身份、split、质量和各族标签mask。检索和迁移直接使用共享z；新增聚类/流形工具只读emb_0..127。模型卡和汇总工具支持分任务报告。
- 修复此前发现的STC默认基线属性错误，以及联合布局中置换检验遗漏response_offsets的问题。

## 修改文件清单

以下是本轮修改/新增文件；工作区原先已有的其他改动不计入本轮交付。

| 类别 | 文件 | 内容 |
|---|---|---|
| 编码与模型 | `src/urban_intervention/representation/encoder.py`、`model.py` | DirectContextEncoder与共享双头模型；复用State编码，保留历史模型 |
| 损失 | `representation/loss.py`、新增`representation/joint_loss.py` | 有限性处理、Alignment有效性统计、分族有效任务归一化 |
| 统一训练 | 新增`representation/training/joint.py` | 三模式训练、裁剪、梯度/尺度诊断、选模、指标与元数据 |
| 两入口接入 | `representation/trainer.py`、`training/runner.py`、`cli.py` | 同一联合实现分派、CLI模式/裁剪/诊断参数；历史路径兼容 |
| 评估 | `representation/evaluation.py`、`baselines.py`、`training/evaluation.py` | 统一布局传递、修复旧STC基线属性访问 |
| 导出 | `representation/export.py` | 新检查点恢复、双头预测/身份/任务mask及schema metadata |
| 实验调度 | 新增`representation/joint_experiments.py` | 三主组及六消融配置、运行状态、代码/数据/城市/cell支持审计 |
| 表示分析 | 新增`representation/embedding_analysis.py` | 仅用共享z128的聚类及UMAP/PCA坐标 |
| 下游工具 | `representation/model_card_cli.py`、`summarize.py`、`scripts/analysis/representation_diagnostics.py` | 联合报告与共享z诊断支持 |
| smoke | 新增`scripts/representation/smoke_joint_training.py` | 合成影像/标签、真实本地冻结DINOv2的工程训练 |
| 测试夹具 | 新增`tests/helpers/joint_stc.py` | 明确non-production的联合标签、缺失场景、PNG和六波段TIFF |
| 测试 | 新增`tests/unit/test_joint_stc.py`、更新`test_modular_representation_trainer.py` | 数值、归一化、梯度、裁剪、无更新、三模式、恢复导出、统计量和下游分析 |
| 环境入口与说明 | `pyproject.toml`、`README.md` | 实验console入口与新架构文档入口 |
| 任务与验收文档 | 新增`docs/operations/stc_joint_training_task.md`、`stc_joint_training_plan.md`、`stc_joint_training_runbook.md`及本报告 | 原任务书合并A–D、计划、验收和正式命令 |

表中`representation/`均相对于`src/urban_intervention/`。既有dataset/collate/publisher已支持事件并集与逐cell掩码，本轮在联合训练入口进一步检查两族布局、角色包哈希、重复cell和城市互斥；没有改写正式数据资产。

原训练器源文件的固定哈希测试改为“两入口必须分派到同一联合实现”的语义检查；历史函数签名/AST一致性和确定性训练产物parity仍执行，避免只更新金标哈希而缺少行为验证。

## 测试与静态检查

最终命令：

```powershell
conda run --no-capture-output -n mit python -m pytest -q -rs --disable-warnings --maxfail=5
```

结果：**814 passed、1 skipped、4 warnings，174.15秒**。唯一跳过项为`tests/unit/test_multimodal_representation.py:207`的真实UMAP验证，需要可选umap-learn。新增表示分析的PCA fallback实际执行。

之后为支持无外网服务器加载本地DINOv2缓存，将加载器调整为本地缓存优先，并增加回归测试。该补丁的本地联合测试为17 passed，Ruff通过；服务器端更新后相关测试为**34 passed、1 warning，27.37秒**。服务器真实加载DINOv2 ViT-S/14成功，22,056,576个参数均被冻结，权重SHA256为`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。

此前针对模型/数据发布/历史STC/导出/评估的指定回归为110 passed。最终全量已包含这些测试及后续增加的共享表示分析测试。

本轮修改/新增代码的Ruff检查通过；全仓扫描仍有此前已有的8个规范问题，位于时序/视觉历史模块和既有审计脚本，本轮未将其混入架构改造。Python源代码编译检查通过。没有运行R参考估计器或新的GPU因果parity，因为本轮没有修改因果估计器。

## 三组主对照运行状态

最终源码使用同一合成发布包、24个事件、相同seed42、每组2 epoch、batch4，在CPU上读取真实冻结DINOv2缓存，消费合成街景及开通前六波段TIFF。

| 实验 | 状态 | 训练优化步 | 表示/条件维度 | 产物 |
|---|---|---:|---|---|
| VIIRS-only | completed | 6 | z128/t32/b96/x256 | checkpoint、配置、loss/梯度日志、测试报告、24事件导出、模型卡 |
| POI-only | completed | 6 | 相同 | 同上 |
| Joint | completed | 6 | 相同 | 同上，含两任务共享梯度和冲突诊断 |

三组使用相同representation代码哈希`105ad37964b7ee87c9444f92fe1350be6e3f44c23c452fe76f6b6d7721337824`，数据bundle哈希、城市划分与两族测试cell支持指纹均一致。测试城市为夹具city_d，VIIRS有效30个cell、POI有效72个cell。

产物在`outputs/stc_joint_20261022/final_smoke/experiments/`；比较状态在`comparison_status.json`。每组均生成任务要求的best_model.pt、training_config.json、training_history.json、test_metrics.json、evaluation_report.json、runs.jsonl，另有gradient_diagnostics.jsonl和embeddings.parquet及其metadata。

## 产物与诊断审计

- 三组均导出24条唯一事件；每条128个emb、256个fused、6个VIIRS预测、12个POI预测。
- z单位范数最大误差小于7.4×10⁻⁸。
- 恢复后导出的预测重新计算测试MSE，与训练评估报告最大差异3.73×10⁻⁹。
- 联合模型两任务对共享State的平均有效梯度范数分别约0.6685、0.8212，均非零；6个诊断batch中3个余弦为负。该计数只证明冲突诊断可运行，不代表真实城市数据中的任务冲突结论。
- 主组实际裁剪前最大范数约3.75；裁剪后双精度重算的最大范数为1.00000140。采用float32计算的1e-5相对容差验收，保留原始测量值，不把浮点累计误差改写为精确1。
- 合成数据中AlphaEarth块RMS约1.0065、表格块RMS约0.999–1.001；两块条件输入梯度均非零。真实背景尺度/贡献仍需在正式数据训练中检查。
- 共享z聚类及二维坐标已导出为`embedding_analysis.json`；本地方法明确为`pca_fallback`，没有冒充UMAP。
- 审计证据：`outputs/stc_joint_20261022/artifact_audit.json`；可复核脚本在同目录`audit_results.py`；三组摘要为`comparison_summary.md`。

## 其他科学消融的工程运行

同日先行的完整smoke套件还完成了Anchor-only、TC-only、z-only、去街景、去卫星、仅表格共6组，每组24事件/2 epoch，全部completed，连同三主组共9组，测试支持审计一致。产物位于`outputs/stc_joint_20261022/smoke/experiments/`。最终三组主对照另用最终源码复验并完成源代码指纹审计；这6组消融属于工程smoke记录，不是正式性能实验。

## 服务器正式验收与运行状态（2026-10-22；最近检查 14:16 CST）

服务器代码暂存于`/workspace/MIT_summer_stc_joint_20261022`，读取原项目正式数据`/home/nas/lsr/MIT_summer/data`。密码仅用于交互式SSH认证，没有写入代码、训练命令、Git或输出日志。DINOv2源码和权重只放入Torch缓存；离线加载补丁避免训练时访问GitHub。

使用服务器现有生产Response Artifact与多模态输入重建联合输入包`data/active/model_inputs/stc-joint-mg-design-v1`。其manifest标记`strict_production=true`、响应族为VIIRS和POI，城市划分为train/validation/test = 31/7/6；含5,048个事件索引、35,366条有效训练标签记录、30,288条卫星时序记录及7,543条街景资产记录。训练入口执行联合schema、包哈希、城市互斥及测试支持检查。

服务器回归测试通过后，于UTC 2026-10-21 19:04启动三组正式对照，等同于上海时间2026-10-22 03:04。配置为seed 42、100 epoch、batch 32、`z/t/b/x=128/32/96/256`、街景和卫星开启、全局梯度裁剪1.0、每批梯度诊断；设备为CUDA。环境将物理GPU 2映射为训练进程的`cuda:0`，不占用当时已有其他进程的前两张卡。

| 实验 | 当前状态 | 已确认 |
|---|---|---|
| VIIRS-only | completed | 100 epoch完成；测试有6个城市、1,233个有效标签cell；RMSE 0.3859、MAE 0.2678、Pearson r 0.1137；相对训练均值基线MSE改善1.86%；共享表示检索 `nn_corr@k=0.0216`，基线0.0033；线性probe RMSE 0.3883 |
| POI-only | completed | 100 epoch完成；测试有6个城市、2,156个有效标签cell；RMSE 0.5843、MAE 0.3062、Pearson r 0.1993；相对训练均值基线MSE改善0.82%；共享表示检索 `nn_corr@k=0.0817`，基线0.0342；线性probe RMSE 0.5910 |
| Joint | running | 约第89/100 epoch（诊断记录最新为epoch 88、batch 78）；State Encoder上VIIRS与POI梯度范数分别约0.3165、0.1227；最近记录的联合梯度范数裁剪前/后为0.6231/0.6231，低于阈值1.0；等待完成训练及测试评估 |

两组单任务结果已生成，联合组尚未完成，因此目前不能判断联合训练是否优于单任务。单任务测试RMSE/MAE为模型标准化目标尺度下的数值；检索指标按各自响应任务计算。比较器完成后还需核对联合组与对应单任务的测试支持指纹，并读取联合组误差、响应检索、近邻一致性和跨城迁移结果。上述性能增益目前较小，作为初步结果报告，不作优越性结论。

### 当前结果的解释

截至联合组完成前，单任务结果整体偏弱，但并非完全没有响应结构信号。VIIRS最近邻响应检索的置换检验p=0.198，200次bootstrap区间为[-0.0062, 0.0488]，包含0；POI检索p=0.0198，区间为[0.0479, 0.1152]，显示出正向关联。跨城线性probe也只有小幅改善：VIIRS RMSE 0.3883，对照训练城市均值0.3894；POI RMSE 0.5910，对照0.6000。六个测试城市间误差波动明显，VIIRS城市RMSE范围0.174–0.484，POI为0.151–0.773。

这些证据支持“表示对构造的响应标签含有一定可迁移关联”，其中POI信号更清楚；它们不足以证明模型已经学到稳定的因果迁移特征。当前评估是按留出城市测试预测的响应标签与表示相似性，标签本身来自既有生产响应估计；VIIRS检索未显著、跨城probe相对均值基线提升很小，且目前只有单seed。应待Joint完成并完成同支持比较、多seed与因果估计敏感性验证后再作因果迁移结论。

训练曲线给出了一个具体的性能限制线索：虽然配置系数为Anchor 0.85、Alignment 0.15，但最后一轮VIIRS训练损失约Anchor 0.027、Alignment 3.289，按系数加权后Alignment约占总目标96%；POI约Anchor 0.054、Alignment 3.027，Alignment约占91%。两组验证Alignment均约6.5，而训练Alignment约3.0–3.3，且从首轮起就有明显差距。这说明实际优化主要由数值尺度更大的Alignment驱动，同时对留出城市的响应几何泛化较差；它比单纯说“轮数不够”更符合当前证据。后续应先用同支持验证调节Alignment尺度/温度或损失归一化，并以Anchor-only及多seed对照检查预测与检索是否改善，再判断因果表示是否更稳。

结果目录为`/home/nas/lsr/MIT_summer/outputs/stc_joint_server_20261022/formal_comparison_seed42/`，主日志为同级`../logs/formal_comparison_seed42.log`，状态文件为`comparison_status.json`。训练进程已脱离SSH会话运行；PID 7787为Conda启动器。首组配置记录代码SHA256 `c473ef76b9bb181cf579e71517d096ae2eb1c4ee7876649a04af3f582ccffd07`。启动时首组状态为running、后两组pending、support_consistent尚待比较器完成后计算。

## 尚未完成与使用边界

1. 服务器Joint正式训练仍在运行（最近记录约第72/100 epoch）；最终联合测试报告、三组支持指纹一致性核对及三组正式结果汇总尚未完成。正式科学消融及多seed稳定性比较也尚未完成。
2. VIIRS-only和POI-only正式评估已完成，联合模型的误差、均值基线提升、响应检索、近邻响应一致性和未见城市泛化结果待训练结束后读取；目前不声称联合训练优于单任务。
3. 本地未安装umap-learn，真实UMAP仍待具备该依赖的环境执行；PCA fallback已验证。
4. 新联合基线固定顶层任务权重和温度，要求queue_size=0；不会静默接受旧的跨batch队列、可学习温度或不确定性自动调权。这些可选扩展不属于本轮固定基线。
5. 服务器已通过相关单测、正式包构建和DINOv2离线加载验收；VIIRS-only与POI-only正式训练、测试评估完成，Joint训练仍在进行。旧模型结果和正式资产未覆盖或删除。

## 正式运行命令

数据发布、三组主比较、科学消融、表示分析及裁剪/诊断参数的完整命令见[运行约定](stc_joint_training_runbook.md)。核心比较入口为：

```bash
python -m urban_intervention.representation.joint_experiments \
  data/active/model_inputs/stc-joint-mg-design-v1 \
  --output outputs/stc_joint/formal_comparison_seed42 \
  --experiments viirs_only poi_only joint \
  --epochs 100 --batch-size 32 --seed 42 \
  --max-grad-norm 1.0 --gradient-diagnostics-interval 1
```

该入口默认要求strict-production数据，并拒绝覆盖已有比较目录。正式报告应同时保留各族有效支持、配置和数据/代码指纹。
