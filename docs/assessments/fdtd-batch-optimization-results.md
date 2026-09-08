# FDTD Batch 前向优化结果与验收

日期：2026-09-08。已实现局部材料编译复用、自动 B、Batch 私有分配池
硬预算、多 GPU/双槽、B/Q/K 吞吐调优、按需电场频谱和原生 checkpoint 伴随。
全计划仍未完成：精确全输出提前规划、融合 B-lane 反向和独占多卡吞吐验收仍待推进。

## 后续功能交付与正确性审计

| 子阶段 | 提交 | 已验收范围 |
| --- | --- | --- |
| R1 密度区域复用 | `237c5b5` | 固定 bank、局部 node/Yee halo、100 次更新与 A/B/A |
| R2 容量与字节预算 | `c5af623` | 单 lane 校准、有界降 B、私有 backing-segment 硬限制与 Graph/保留结果计费 |
| R3 多 GPU/双槽 | `4c8f159` | 有界队列、静态/吞吐调度、取消、跨卡输出和调用流同步；118 项组合通过 |
| R4 Batch 伴随 | `829db6a` | 原生批量前向及逐 lane GPU 反向；密度/源/共享参数、双卡和延迟反向；80 项组合通过 |
| R9 Result 契约 | `d12e001` | 场景声明快照、原始梯度版本保护、公开 tensor/field 和保存/加载；80 passed |
| R7 吞吐调优 | `b902bf5` | 完整流式前向候选测量、B/Q/K 选择、结果/RNG 隔离、正式上下文重建；双卡组合 25 passed |
| R8 按需频谱 | `c4a71a7` | 只分配请求分量、独立全场对照、有限差分；数值/普通伴随 96 passed、4 skipped，后续显式启用原生专项 22 passed |
| R5 结构区域复用 | `96eab31` | Box/Sphere 包络、材料变化与失效检查；区域/benchmark 回归 114 项通过 |

最终集成修复把包络放到 worker 的设备上，避免在另一 GPU 的私有池上下文中
分配证明用的临时张量。每个可求导 Result 记录检查点个数与逻辑字节数，
反向后记录同一分配账户的 charged/peak bytes。快照、Graph 缓存及保留的
输出持续计费；CUDA context 和库内部非 allocator 分配不在该硬上限内。

反向正确性限定：支持密度、非磁非导电介质结构几何及源幅度梯度。训练密度
改变 mu_bounds、训练几何改变磁性/导电/PEC 分布，以及未注册的材料区域几何
输入会明确拒绝，避免只有部分物理导数。torch.no_grad 可执行前向；重新启用
训练会重新验证。源归一化只移除单位幅度包络，保留物理幅度的导数。
PyTorch 在双槽旧图共享叶子时可能提示 AccumulateGrad stream mismatch；
梯度对照通过，但额外同步仍是反向性能改进项。

最终集成矩阵 115 passed，包含实际双卡、预算、材料编译和梯度；测试
数量有重叠，不累加。静态调度的末任务领取竞态及最后发布超时竞态均修复
并有确定性覆盖，随后的调度专项 13 passed。Batch 使用线程局部捕获错误
检查，独立消费线程在 capture 期间分配/计算的实测及 Graph 回归 22 passed；
普通 Graph runner 保持原来的全局模式。原输入和执行槽仍须遵守所有权约束。
缓存另覆盖参数字典重排、requires_grad 切换和 no_grad/训练切换，CPU 导出
也在参数更新时重新检查其不可求导契约。

## 局部材料编译的新增测量

同一空闲 A6000、2³ 偏振子像素、2 次预热/5 次交错正式测量：

| 节点网格 | 变化 | 完整编译中位 s | 局部编译中位 s | 倍速 |
| --- | --- | ---: | ---: | ---: |
| 57³ | 密度 | 0.286664 | 0.288384 | 0.994 |
| 57³ | 结构/材料 | 0.142815 | 0.145700 | 0.980 |
| 137³ | 密度 | 0.594244 | 0.270486 | 2.197 |
| 137³ | 结构/材料 | 0.293651 | 0.142700 | 2.058 |

所有材料张量误差为零。小网格受 Python/CUDA 提交开销主导，裁剪体积
未减少操作数量；大网格减少的采样工作带来实测收益。以上不包含初始 bank、
solver 构建和 FDTD 时间推进，不能当作端到端加速比。原始数据与可复现
入口见 benchmark/plots/batch/regional-material-reuse*.json 和
python -m benchmark batch --regional-audit local_spectrum（或 bandwidth）
--warmups 2 --repeats 5 --update-report。

## 已交付子阶段

| 子阶段 | 提交 | 实现与验收依据 |
| --- | --- | --- |
| O1 观察器批处理 | `4e3922a` | 点/面/通量频谱跨 lane 和频点累积；相位、窗、源归一化状态由 GPU 推进；35 项专项通过 |
| O2 去除未启用材料分配 | `97a3a57` | 未启用通道使用标量广播；空色散模板不分配整域逆材料；材料/色散/梯度矩阵 175 passed、1 skipped |
| O3 源与完整时间循环 | `f7c7b71` | 源 patch 跨 lane 注入，保留重叠顺序和周期端点；普通 generator 只负责初始化/收尾；46 项回归通过 |
| O4 系数共享 | `7de393a` | 编译值相等才共享；后续差异单向升级为逐 lane 缓冲；地址与数值回归通过 |
| O5 完整时间步 Graph | `8f81a0e` | 场、CPML、源、边界、DFT、观察器统一 capture/replay；尾块、提前停止、重复运行和 capture 协调验证通过 |
| O6 固定模板材料复用 | `3e123d7` | 只有源幅度变化时跨 cohort 共用一次材料编译；N=5/B=2 更新、密度隔离与生命周期共 20 项通过 |
| O7 静态材料数组共享 | `8b65420` | 每个 lane 构建后立即共享实际相等的只读数组；不同介电区域保持独立；27 项数值/生命周期回归通过 |

子阶段测试存在重叠，不相加作为唯一测试数量。最终组合回归记录见文末。

## 性能证据

设备为物理 GPU 1 的 RTX A6000，映射为 `cuda:0`；环境为 `maxwell`，
PyTorch 2.13.0+cu130。GPU 0 的其它任务未修改，不声称整机独占。
计时是准备后复用 context/process 的完整暖态轮次，包含 CPU 频谱可用前的等待；
不是只计 kernel，也不是首次编译、启动或 capture 的耗时。
驱动显存使用独立新会话和 20 ms 采样，可能遗漏短暂峰值。

小网格：16-cell，N=B=10，2000 步，2 次预热、5 次正式测量。

| 执行方式 | 中位耗时（秒） | 隔离驱动显存增量（MiB） |
| --- | ---: | ---: |
| 顺序 | 3.615720 | 315.5625 |
| 十进程 | 0.556994 | 3157.6250 |
| run_many | 10.742862 | 351.5625 |
| Batch 完整 Graph | 0.126706 | 401.5625 |

Batch 比十进程快约 **4.40 倍**，采样驱动增量少约 **87.3%**。
所有对照频谱探针误差为零。仅此负载的组合耗时/显存门通过，不能外推到
原始业务几何或完整 N/B/输出矩阵。
原始数据：`benchmark/plots/batch/optimized-launch-n10-graph.json`。

31 频点诊断：48-cell、B=10、64 步、eager，最终暖态中位 0.05369 秒。
点谱 gather 从原来的 19,840 次变成 64 次，源注入从 640 次变成 64 次，
最终记录的 CUDA activity 总数为 817。观察器 CPU 区间约 6.55 ms，
占当次 66.33 ms profiling 区间约 9.9%，原审计约为 90.3%；关联 GPU
观察器工作约 0.54 ms。原始记录为 `optimized-pipeline-spectrum.json`。
跨会话暖态数字不作为受控倍速结论，调用次数下降则直接验证了批处理效果。

大网格存储诊断：128-cell、B=4、64 步、eager，槽可达张量由原审计的
约 2677.4 MiB 降至 1128.1 MiB，减少约 **58%**。
最终暖态约 0.146 秒；相等材料共享前后耗时基本不变。
约 98.5% 的 GPU 时间仍在 E/H 场更新，场 Graph 单独消融的收益很小。
原始数据：`optimized-bandwidth-b4.json`、`optimized-bandwidth-shared-b4.json`，
均位于 `benchmark/plots/batch/`。这些是可达张量指标，不能替代驱动峰值。

大网格完整对照：128-cell，N=B=4，256 步，2 次预热、3 次正式测量。

| 执行方式 | 中位耗时（秒） | 隔离驱动显存增量（MiB） |
| --- | ---: | ---: |
| 顺序 | 0.637905 | 859.5625 |
| 四进程 | 0.729172 | 3438.2500 |
| run_many | 0.656874 | 2513.5625 |
| Batch 完整 Graph | 0.571042 | 2409.5625 |

Batch 比四进程快约 **1.28 倍**，采样驱动增量少约 **29.9%**，
此负载的组合门通过。每个 lane 立即共享材料使 Batch 驱动增量从中间版本的
2849.6 MiB 降至 2409.6 MiB；准备瞬时峰值解释了驱动增量与 1128.1 MiB
驻留张量之间的差距。原始对照为 `optimized-bandwidth-n4-graph.json` 和
`optimized-bandwidth-incremental-n4-graph.json`，位于 `benchmark/plots/batch/`。
不能由这两次会话推断精确的准备耗时变化。

## 正确性与资源生命周期

- 所有演化场、CPML 历史、观察器累积量保持逐 lane 独立；只有通过实际
  dtype/device/shape/value 检查的只读材料和系数才共享。
- 点源内点、周期面/边角、重叠源及 CW/Gaussian 注入有逐位对照。
  开发中发现普通乘加改变舍入，已显式恢复原生 FMA/周期独立乘法语义，
  没有放宽原有数值容差来掩盖差异。
- Graph 在 planned end 和 shutoff 检查点切块，尾块不越界；lane 停止后
  active mask 阻止继续写入其状态。重复运行重置波形、相位、窗及累积状态。
- R11 使 monitor-only 和所选全场 DFT 都在同一驻留 cohort 中复用缓冲和 capture；
  标量波形的固定采样表也复用。Result 使用独立存储，close/下一轮
  不会改写旧结果。
- 源幅度模板的材料复用只适用于固定模板、无密度参数和动态/可训练材料。
  密度变化仍独立编译。输入身份/version/inference tensor 的既有缓存保护保留。
- 换出 cohort 在关闭循环垃圾回收时仍及时释放。去重递归使用模块函数，
  避免递归闭包形成环并滞留旧 solver；旧 graph/source/observer 在构建下一批前释放。
- capture 协调器保护已注册 Batch 线程及现有 capture suspension 执行器。
  auto 无法安全捕获时报告 eager/fallback reason，required 明确失败。
  Batch 使用 thread_local 捕获检查，外部线程可以操作独立 GPU 张量；不能修改捕获输入和槽存储。

上述主要是普通求解器兼容性和底层数值回归，不能替代独立物理验证，
也不代表 Batch 伴随已验收。

## 使用方式

```python
import witwin.maxwell as mw

batch = mw.Simulation.batch(
    simulations,
    # Include this declaration when the named source changes between inputs.
    variation=mw.BatchVariation(sources=("drive",)),
    execution=mw.BatchExecution(
        devices=("cuda:0",), batch_size=4,
        graph="required", graph_steps=32,
    ),
    outputs=mw.BatchOutput(monitors=("probe",), fields=(), result_device="cpu"),
)
with batch.prepare() as prepared:
    result = prepared.run()
    spectra = result.stack(lambda r: r.raw_monitor("probe")["components"]["Ez"])
    report = prepared.report.to_dict()
```

按实际源/监视器名字调整声明。`batch_size="auto"` 已使用单 lane 实测
保守选择 B，并在发布结果前对容量失败有界缩小重试。内存较紧时流式消费结果。
需要自动退回 eager 时使用 `graph="auto"`，检查实际 report。

## 剩余瓶颈和未验收范围

1. **大网格场更新带宽**：每个变体的 E/H 和 CPML 必须保留独立状态，
   局部几何变化不意味着场变化局限在该区域。进一步收益需要实测指导的
   场 kernel 融合、访存优化或已证明安全的系数压缩，不能直接共享演化场。
   尚无硬件带宽计数器证据，不声称已达到显存带宽极限。
2. **N>B 的准备成本**：固定 bank 和局部 node/Yee 编译已复用，但换入
   仍构建普通 solver。小网格的局部编译仍受 Python 提交开销限制；通用复杂
   几何和扰动网格未接入局部证明。
3. **预算与调优**：硬限制按私有 allocator backing segments 计费，保留输出、
   tape 和反向 workspace 不能无限累积。驱动 context/库内分配不在池内；
   精确全输出提前规划与自动吞吐调优尚未实现。多 GPU/双槽已验证功能，
   GPU 0 有其他任务，不能据本次测试验收独占设备的扩展比。
4. **输出及训练**：请求某个 E 分量时仍分配普通全 E 累积器。磁场全域谱、
   材料导出、RF 和高级动态材料覆盖仍有限。Batch 反向已可执行，当前逐 lane
   复用原生伴随，尚未融合反向 kernel；检查点与不可变快照增加训练显存。

完整阶段、验收标准及未完成契约以
[开发方案](../plans/fdtd-batch-simulation-api-and-gpu-scheduling.md) 为准。

## 先前前向阶段的组合验收

实现基线 `8b65420`：**353 passed，0 failed，0 skipped**，204.77 秒。
只有 benchmark 稀疏 storage manifest 测试的一条既有 PyTorch invariant warning。
在 `maxwell` 环境、物理 GPU 1 上运行以下组合；不是全仓库全量测试。

```bash
python -m pytest \
  tests/api/public/test_batch*.py \
  tests/fdtd/cuda/test_batch*.py tests/fdtd/cuda/test_cuda_batch.py \
  tests/fdtd/test_batch*.py \
  tests/materials/compiler/test_batch_material_compiler.py \
  tests/materials/compiler/test_inactive_material_storage.py \
  tests/fdtd/cuda/test_cuda_graph.py tests/fdtd/test_cuda_graph.py \
  tests/boundaries/cpml/test_fdtd_cpml.py \
  tests/monitors/observers/test_fdtd_observers.py \
  tests/gradients/test_fdtd_adjoint_bridge.py \
  tests/gradients/test_fdtd_adjoint_materials.py \
  tests/validation/benchmark/test_batch_benchmark.py -q
```

性能验收复现入口：

```bash
python -m benchmark batch --compare launch --variants 10 --concurrency 10 \
  --include-native --native-graph required --warmups 2 --repeats 5
python -m benchmark batch --compare bandwidth --variants 4 --concurrency 4 \
  --steps 256 --include-native --native-graph required --warmups 2 --repeats 3
python -m benchmark batch --profile local_spectrum --concurrency 10 \
  --steps 64 --warmups 2 --repeats 5
```

实际硬件映射与 toolkit 环境变量见
[实施证据](fdtd-batch-implementation-progress.md)。未请求修改其它设备或系统配置。

## R7：完整前向吞吐调优

已实现 `tune=True` 的 B/Q/K 有限候选实测，隔离预热一次、计时三次，选择时考虑显存和测量波动。真实双 GPU 调优/容量/调度组合 25 passed；公共 API/调优组合 48 passed。测量包含全部前向与输出导出、训练模式的 tape 创建，排除 backward 和用户保留全部结果。它提供当前运行环境下的候选证据，不承诺消除其他进程造成的噪声；物理 GPU 0 仍有其他工作，本轮未据此验收双卡 1.7 倍吞吐门槛。

## R8：按请求分量分配 DFT

只请求 Ez 不再分配 Ex/Ey 的全域谱累积器。实场的该部分容量为 `8 F Σ numel(component)`，统计直接由实际张量计算。原生 CUDA 各 launch 分支保留原公式；省去未请求分量的读写和最终谱构造，伴随种子也不再构造无用分量。数值/普通伴随回归 96 passed、4 skipped；这提供存储和正确性证据，不单独宣称业务端到端加速倍数。

## R9：结果描述和公开字段契约审计

结果场景现在保存独立的无梯度声明快照，修复输入密度或几何更新后旧 Result.scene 跟着变化的问题；快照存储进入执行设备的 Batch 分配账户。伴随仍使用原始输入及版本保护。另修复全场导出大小写不符合普通 Result.tensor/field 查询规则的问题，专项改用公开接口验收。快照与前向/梯度/真实双卡等组合 61 passed，有一个已知流同步提示。

R9 最终公开契约回归：80 passed，1 个已知 AccumulateGrad 流提示。覆盖 CPU 导出及保存/加载后通过 tensor("Ez") 访问；未保留旧的非规范字段键。

## R10：十设计实测与调优成本

统一 benchmark 的 launch 调优审计（N=10、2000 步）记录 30 个候选。选中 B=10/Q=1/K=16，为候选中最快，中位 0.124765 s；额外 5 次热测量中位 0.130684 s，独立探针误差 0。候选私有池峰值 58 MiB，不能当作驱动显存。调优总成本 77.096 s；默认步长 K=32 的同容量候选 0.131157 s，因此这次测量不支持“调优优于默认配置”的结论。一次性短任务应考虑直接保留 tune=False。Q=2 自动选择门未触发。

JSON：benchmark/plots/batch/tuned-launch-n10.json；所有样本和环境写入其中。报告/CLI/benchmark 系统组合 111 passed、3 skipped。实际 GPU 0 仍运行其它工作，未提交独占多 GPU 的吞吐倍数结论。
