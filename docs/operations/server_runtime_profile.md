# 生产服务器运行配置

本文件记录全量因果标签生产使用的服务器与容器资源。采集时间为 2026 年 8 月 31 日，项目目录为 `/home/nas/lsr/MIT_summer`。

## 1. 处理器与容器配额

| 项目 | 实测值 |
|---|---|
| 处理器 | AMD EPYC 7402 24-Core Processor |
| 物理核心 | 24 |
| 逻辑处理器 | 48 |
| 插槽 | 1 |
| 每核心线程 | 2 |
| 最大频率 | 2.8 GHz |
| L3 缓存 | 128 MiB |
| NUMA 节点 | 1 |
| 容器可用处理器 | `0-47` |
| cgroup CPU 配额 | `max 100000`，未限制总 CPU 时间 |

容器能够使用全部 48 个逻辑处理器。单一 NUMA 节点使进程不需要额外设置跨节点内存亲和性。

此前以 4 个控制设计 worker 运行时，每个 worker 约占用一个逻辑处理器，单进程 CPU 使用率接近 100%。同期系统 CPU 空闲率约为 91% 至 92%，I/O 等待接近 0%。这组观测支持继续测试 8 和 16 个 worker。

## 2. 内存

| 项目 | 实测值 |
|---|---|
| 总内存 | 503 GiB |
| 已使用 | 17 GiB |
| 可用 | 486 GiB |
| Swap | 2.0 GiB |
| 已使用 Swap | 469 MiB |
| cgroup 内存上限 | `max` |

当前内存容量足以支持提高控制设计并发度。扩大并发时仍需记录单个 worker 的峰值常驻内存和全局可用内存。

## 3. GPU 与 CUDA

| 项目 | 实测值 |
|---|---|
| GPU | 4 × NVIDIA GeForce RTX 4090 |
| 单卡显存 | 24,564 MiB |
| 单卡功耗上限 | 450 W |
| 驱动版本 | 580.173.02 |
| 驱动支持的 CUDA 版本 | 13.0 |
| PyTorch | 2.11.0+cu130 |
| PyTorch CUDA | 13.0 |
| PyTorch 可见设备 | 4 |
| Compute Mode | Default |

空闲状态下四张 GPU 均可见。此前 4 个控制设计 worker 分别连接 GPU 0 至 GPU 3，单进程显存约为 0.8 GiB 至 0.95 GiB。控制设计阶段包含较多 CPU 面板构建工作，因此瞬时 GPU 利用率不能单独代表管线吞吐量。

## 4. 存储与网络

| 项目 | 实测值 |
|---|---|
| 项目存储 | `222.29.102.38:/mnt/Data/share` |
| 挂载点 | `/home/nas` |
| 文件系统 | NFS 4.2 |
| 总容量 | 200 TiB |
| 已使用 | 161 TiB |
| 可用 | 40 TiB |
| 使用率 | 81% |
| NFS 读块大小 | 1,048,576 bytes |
| NFS 写块大小 | 1,048,576 bytes |
| 传输协议 | TCP |
| 挂载策略 | `hard,timeo=600,retrans=2,noatime` |
| 客户端地址 | `222.29.102.59` |
| 网络接口 | `eth0` |
| 链路速率 | 10,000 Mb/s，全双工 |

10 Gb/s 是网络接口协商速率，对应约 1.25 GB/s 的理论上限。生产目录的实际吞吐还取决于 NFS 服务端、共享负载、请求大小和元数据操作。并发决策需要使用 `fio` 实测结果和真实控制记录完成速率共同判断。

## 5. 并发测试基线

| 指标 | 4 worker 观测 |
|---|---|
| worker CPU 使用率 | 每个约 99.7% 至 99.8% |
| 全局 CPU 空闲率 | 约 91% 至 92% |
| I/O 等待 | 接近 0% |
| 完成速率 | 约 12.1 个控制记录每分钟 |
| 单 GPU 进程显存 | 约 0.8 GiB 至 0.95 GiB |

2026 年 8 月 31 日的扩展并发观测如下：

| 并发度 | 可运行进程 | 用户态 CPU | 系统态 CPU | CPU 空闲 | I/O 等待 |
|---:|---:|---:|---:|---:|---:|
| 8 worker | 8 | 16% 至 17% | 0% 至 1% | 83% | 0% |
| 16 worker | 16 | 32% 至 33% | 1% | 66% 至 67% | 0% |

两档并发的可运行进程数与 worker 数一致，CPU 使用率近似线性增加，未观察到 I/O 等待。真实任务当时主要使用页缓存，`vmstat` 没有记录持续块设备读取。

正式调优依次比较 4、8、16 个 worker。每档至少运行 10 分钟，并记录以下指标：

1. `control_record.csv` 的新增数量和每分钟完成速率。
2. `vmstat` 的 CPU 空闲率和 I/O 等待率。
3. `pidstat` 的单进程 CPU 与读写速率。
4. `nfsiostat` 的 NFS 吞吐、平均 RTT 和执行时间。
5. 四张 GPU 的显存、利用率和功耗。
6. worker 失败数、临时文件数和队列状态。

当 16 个 worker 的完成速率相对 8 个 worker 有稳定提升，I/O 等待低于 5%，任务没有新增错误时，可将 Phase 1 设为 16 个 worker。Phase 2 的 `shard-count` 保持为 4，使四个标签分片分别使用一张 GPU。

## 6. 待补充测量

NFS 基准使用 4 GiB 临时文件、直接 I/O 和 `fio 3.36`，采集时间为 2026 年 8 月 31 日。

| 测量 | 结果 |
|---|---|
| NFS 顺序写带宽 | 25.1 MiB/s，26.3 MB/s |
| NFS 顺序写平均延迟 | 159.26 ms，4 MiB 请求 |
| NFS 顺序读带宽 | 108 MiB/s，113 MB/s |
| NFS 顺序读平均延迟 | 37.07 ms，4 MiB 请求 |
| NFS 四任务并行读带宽 | 112 MiB/s，118 MB/s |
| NFS 四任务并行读 IOPS | 112，1 MiB 请求 |
| NFS 四任务并行读平均延迟 | 284.61 ms |
| NFS 四任务并行读 P95 延迟 | 330 ms |
| NFS 四任务并行读 P99 延迟 | 355 ms |
| 8 worker 控制记录每分钟 | 约 26.3，15 分钟样本 |
| 16 worker 控制记录每分钟 | 约 18.2，15 分钟样本 |
| 24 worker 控制记录每分钟 | 约 26.5，启动后 5.8 分钟样本 |

四任务并行读带宽只比单任务顺序读高约 4%，说明当前共享存储的聚合读取平台约为 110 MiB/s。并行请求主要增加排队延迟。写入带宽约为读取带宽的四分之一。正式并发度以真实控制记录完成速率为主，并同时观察 NFS 延迟和 I/O 等待。

三档 worker 吞吐测试处理了不同的剩余任务集合，城市、donor 数量和数据支持并不完全相同。测试期间还出现过控制记录总数非单调变化。因此这些结果用于判断并发平台，不作为严格的等任务基准。当前证据显示超过 8 worker 后没有稳定的线性加速。

## 7. Phase 1 完成快照

2026 年 9 月 1 日，正式控制设计队列达到完整终态：

| 状态 | 数量 |
|---|---:|
| `matched` | 3,032 |
| `gsc_pending` | 2,016 |
| 总行数 | 5,048 |
| 唯一 `treatment_order` | 5,048 |
| 非终态 | 0 |

`gsc_pending` 是 Phase 1 的合法终态，表示固定对照没有通过预处理期匹配或 placebo 质量门槛，Phase 2 将按正式路由继续尝试 GSC 和 MC；它不表示 Phase 1 执行错误。此次快照之后新增的按 outcome 独立抽样、Response Artifact 子集发布和 event-study/汇总过滤代码不参与 Phase 1 控制设计计算，因此不改变上述 5,048 条控制设计结果。

## 8. 配置复核命令

```bash
lscpu
nproc
numactl --hardware
grep Cpus_allowed_list /proc/self/status
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/cpuset.cpus.effective
free -h
cat /sys/fs/cgroup/memory.max
nvidia-smi --query-gpu=index,name,memory.total,power.limit,compute_mode --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
df -hT /home/nas/lsr
findmnt -T /home/nas/lsr -o TARGET,SOURCE,FSTYPE,OPTIONS
nfsstat -m
```
