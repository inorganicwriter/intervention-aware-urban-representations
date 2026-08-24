# 模块化因果标签队列

本目录按职责拆分因果标签队列。规范生产入口仍为
`scripts/causal_python/run_causal_label_queue.py`；模块化入口为：

```powershell
python scripts/causal_python/run_causal_label_queue_modular.py --help
```

模块边界：

- `runtime.py`：进程级设置和共享路径；
- `state.py`：队列、任务路径、子进程及规格指纹；
- `support.py`：结果族和 VIIRS 支持判断；
- `validation.py`：标签、manifest、恢复和失效校验；
- `matching.py`：同城/跨城 Matching 与固定控制标签；
- `estimators.py`：Python 估计器共同合同和结果规范化；
- `gsc.py`、`mc.py`：各自后端；
- `orchestrator.py`：Matching → GSC → MC → skip 路由；
- `cli.py`：参数、shard 和任务分派。

`tests/unit/test_modular_causal_label_queue.py` 检查原文件哈希、函数结构、CLI
参数、规格指纹、输出路径、任务筛选、dry-run 和 fallback 路由。模块化入口
不用于正式生产。
