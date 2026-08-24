# 模块化项目配置

本目录按职责拆分项目配置。现有调用方仍导入
`urban_intervention.config.project`；模块化门面为
`urban_intervention.config.project_modular`。

- `registry.py`：44 城市注册、活动城市及地铁参考；
- `pipeline.py`：采集和数据管线参数；
- `filesystem.py`：项目路径别名与目录初始化；
- `boundaries.py`：行政边界、bbox 和网格裁剪；
- `stations.py`：站名规范化兼容函数；
- `network.py`：显式代理检测、缓存和覆盖。

`tests/unit/test_modular_project_config.py` 检查公开常量、函数签名、函数结构、
城市 bbox、边界 fallback、站名和代理状态。模块化门面不用于正式生产。
