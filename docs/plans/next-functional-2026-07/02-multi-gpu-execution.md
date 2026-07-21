# Multi-GPU Execution Infrastructure Plan

> Status: proposed  
> Date: 2026-07-14  
> Target evidence: ensemble E3; joint-solve forward E3, adjoint E2  
> Depends on: stable `Simulation` / `Result` contracts  
> Owner modules: `simulation.py`, new execution package, `fdtd/`, compiler and result persistence  
> Latest architecture decision: 2026-07-14，ensemble execution 与 joint solve 使用显式不同 strategy，共享设备/诊断层但不共享物理状态  
> Detailed joint-solve precursor: [FDTD multi-GPU implementation plan](../fdtd-multi-gpu-implementation-plan.md)  
>
> **Round-E revision (2026-07-21, master `6b523b8`).** S4 distributed CPML-trainable
> adjoint landed (`2364533`/`f7e8e9a`): psi-carrying reverse with no psi halo, public
> validator relaxed to accept `cpml`/`stablepml`, psi-active 1-vs-2-GPU gradient parity
> rel 5.94e-7 with a ~1.1e5× falsification, after fixing a pre-existing single-GPU CPML
> psi axis cross-wiring (`a2d2cb7`). Forward monitor gather with a seam-ownership rule +
> defense-in-depth trainable guard delivered. Exclusive-window timing measured (ensemble
> 1.98–2.00×; joint-solve forward 128³ 0.544× / 192³ 1.726×; NCCL step-rate
> not-measurable-by-hooks). Evidence: forward E2, CPML-trainable distributed adjoint E2.
> **Still open (not completed):** NCCL one-process-per-GPU adjoint + monitor gather and
> coupled-runtime (circuit/network/wire) joint solve remain fail-closed (blueprint
> #13/#18); no non-author review. See `docs/assessments/e3-distributed-adjoint-acceptance-2026-07-19.md`,
> `multi-gpu-timing-2026-07-20.json`, `00-status-and-gaps-2026-07-19.md` §02.

## 1. 功能定位

多 GPU 是横向执行基础设施，而不是一个单一加速开关。Maxwell 需要明确支持两种不同工作负载：

1. **Ensemble multi-GPU**：多个彼此独立的 `Simulation` 分配到不同 GPU，服务 N-port 逐端口激励、频率/参数扫描、Monte Carlo 和阵列 codebook；
2. **Joint-solve multi-GPU**：一个 FDTD 场景按空间域分解到多个 GPU，服务单卡显存无法容纳的大网格，并在足够大问题上获得加速。

两种模式共享设备拓扑、调度、错误传播、统计和结果顺序语义，但 ensemble 不交换场，joint solve 必须在每个 E/H half-step 交换 Yee halo。不得以复制完整场到每张卡的 data parallel 假装完成 joint solve。

## 2. 当前状态

- `Scene.device`、`PreparedScene`、FDTD solver 和 Result field placement 都面向单个设备；
- FDTD 六个 staggered field、材料系数、CPML/ADE state、DFT、observer 和 adjoint checkpoint 是全局 tensor；
- CUDA Graph capture 是单设备计算序列；
- repository 尚无 device pool、公共批执行器、shard layout、halo transport 或 distributed Result manifest；
- 已有独立 joint-solve 设计详细规定 x-slab ownership、P2P transport、monitor-first gather 和 adjoint reverse halo，本计划将其纳入统一路线，不另造第二套域分解。

## 3. 目标与非目标

### 3.1 目标

- 保持每个物理求解仍是 `Scene -> Simulation -> Result`；
- 为一组 Simulation 提供稳定、有序、失败可追踪的多 GPU ensemble 执行；
- 让 joint solve 的可解网格规模近似随总显存增长；
- 对足够大网格重叠 halo 通信与 interior update；
- 保留单 GPU 默认行为和性能；
- 所有正常 hot path 保持 device resident；
- 给每个新 runtime feature 一个统一 distributed contract；
- 为 PyTorch adjoint 提供可推导的 reverse ownership 和 collective 规则。

### 3.2 第一生产里程碑不包含

- FDTD 以外的求解器域分解；
- 异构 GPU 动态负载平衡；
- CPU/host-staged halo fallback；
- 自动 2D/3D decomposition；
- 跨节点 MPI；
- 通信参与的完整 CUDA Graph capture；
- 所有高级材料在第一版同时获得 multi-GPU adjoint。

## 4. 公共 API 草案

### 4.1 Ensemble execution

保持 Simulation 为工作单元，增加执行器而不是第二物理入口：

```python
import witwin.maxwell as mw

execution = mw.MultiGPUExecution.ensemble(
    devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    max_concurrency=4,
    placement="memory_aware",
    fail_fast=False,
)

results = mw.run_many(simulations, execution=execution)
```

契约：

- 输入和输出严格同序；
- 每个 Simulation 仍单独产生 Result；
- scheduler 在 prepare 前估算显存，不在 OOM 后静默迁移；
- 单任务失败返回带原始 index/device/exception 的结构化错误；
- `fail_fast=False` 允许其他独立任务完成，但不吞掉异常；
- trainable simulations 可以独立 backward，不跨任务隐式求和；
- N-port 或 sweep 上层可以使用该执行器，但不得把电磁语义写入通用 scheduler。

### 4.2 Joint solve

```python
parallel = mw.FDTDParallelConfig(
    devices=("cuda:0", "cuda:1"),
    decomposition_axis="x",
    transport="auto",
    overlap=True,
    gather_fields=False,
)

simulation = mw.Simulation.fdtd(scene, frequencies=freqs, parallel=parallel)
result = simulation.run()
```

契约：

- `parallel=None` 使用现有单 GPU runtime；
- 第一版只接受 homogeneous NVIDIA GPUs 和 x-slab；
- 不支持 direct peer transport 时明确失败，不静默经过 CPU；
- 默认只归约 monitor/metadata，不把完整场汇聚到 GPU 0；
- `gather_fields=True` 必须显式请求并在执行前验证目标显存；
- L3 之前 trainable scene 明确拒绝，不能 detach 后继续；
- `Result.solver_stats` 记录 topology、partition、transport、halo bytes、communication/compute time 和各卡 peak memory。

最终命名在 API review 中冻结；两种策略不能共用一个含糊的 `multi_gpu=True` 布尔值。

## 5. 内部设计

### 5.1 共享执行层

引入小而明确的内部对象：

- `DevicePool`：设备发现、能力/显存/topology 快照和租约；
- `ExecutionPlan`：不可变任务顺序、placement 与预估资源；
- `ExecutionTopology` / `DomainPartitionPlan`：与具体 PDE 无关的 rank、邻接、全局/局部 extent 和 collective 计划；
- `DistributedResultManifest`：全局顺序、shard artifact、归约 provenance 和恢复信息；
- `ExecutionRecord`：task/device/timing/memory/error；
- `ResultSequence`：保持输入顺序的 Result 容器；
- `DistributedFailure`：保留 task index、rank/device 和原始异常链。

共享层不得知道 Yee field、端口或热/静电 stencil 语义。Ensemble 使用 device lease；joint solve 使用固定 topology reservation。`FDTDPartitionPlan/FDTDShardLayout`、后续 thermal partition 和 electrostatic partition 都是通用 topology/partition contract 的 runtime specialization，而不是互相复用物理 halo 顺序。

### 5.2 Ensemble scheduler

- prepare 与 run 都在目标设备上下文中执行，避免先在 coordinator GPU 创建完整 PreparedScene；
- 每张 GPU 同时默认运行一个大型 solver；只有测量证明安全时才允许 oversubscription；
- 使用线程/进程的选择由 CUDA context、编译扩展和 Windows/Linux 行为基准决定，不预先锁死；
- result 可按各 Simulation 原始 device 保持，或根据显式 `result_device` 搬运；
- scheduler 不轮询 `.item()`；完成通知不得让所有 device 全局同步；
- 对相同 immutable scene 的多个激励，后续可以缓存 CPU-side scene description，但不能共享可变 solver state。

### 5.3 Joint-solve partition

复用现有域分解设计：

- `FDTDPartitionPlan` 保存全局 x-cell half-open partitions；
- `FDTDShardLayout` 保存各 Yee component 的 owned/halo slices、global offsets 和 physical-face ownership；
- `FDTDShard` 保存本地 fields、coefficients、state、streams、events、sources 和 observers；
- `HaloTransport` 定义 preflight、exchange、reduction 和 teardown；
- `CudaP2PHaloTransport` 是 Windows/单进程首个 transport；
- `NcclHaloTransport` 是 Linux 单进程/多进程后续 transport；
- `DistributedFDTD` 组合 shard-local primitives，不把 device list 扩散进现有 `FDTD` 每个方法。

当前 `(x, y, z)` 存储使 x-normal `(y, z)` plane 连续，因此第一版选择 x-slab，最多两个邻居且无需 packing kernel。

### 5.4 Yee ownership 与 half-step

- `Ex/Hy/Hz` 按 x-cell interval 所有；
- `Ey/Ez/Hx` 按 low-node interval 所有，末 shard 额外拥有终端 node plane；
- interface 值只有一个 owner，ghost 只读且不进入 Result/monitor 双计数；
- H half-step 前交换所需 `Ey/Ez`，E half-step 前交换 `Hy/Hz`；
- interior kernel 与通信并行，只有 boundary strip 等待对应 receive event；
- internal face 永远不能运行 PEC/PMC/PML/Mur 等物理边界逻辑；
- periodic/Bloch 只在 global wrap 建 ring neighbor，Bloch phase 只施加一次。

### 5.5 编译器和状态分片

L1 生产版本必须直接编译 local tensors，不能先在 GPU 0 编译全局 tensor 再切片：

- local Yee coordinates 由 global grid slice 派生；
- geometry/material rasterization 只覆盖 local extent 与 subpixel margin；
- CPML、ADE、nonlinear、modulation 和 SIBC state 按 cell/surface owner 分配；
- point source/monitor 只有一个 owner；plane/volume object 分 tile；
- adjacent shards 的 interface material 编译结果必须一致；
- trainable tensor lineage 不能经过 NumPy 或 CPU 转换。

### 5.6 Monitor、Result 和持久化

- scalar monitor 使用确定性 device-side reduction；
- plane/mode monitor 按全局坐标排序 concat；
- full-field DFT 保持 sharded；
- 大结果使用 manifest 加每 shard tensor 文件保存；
- 绘图只 gather 请求 slice；
- ensemble Result 保持 task 顺序，joint Result 保持空间全局顺序；
- public Result 不暴露 raw CUDA pointer、rank-local mutable state 或 transport handle。

### 5.7 PyTorch 与 adjoint

Ensemble 模式中每个 Simulation 保持现有单卡 autograd graph。Joint solve 的 L3：

- checkpoint 保存 owned state 与 partition manifest，不保存 transient receive halo；
- replay 后重新交换 halo；
- reverse pass 使用 forward ownership/exchange 的转置；
- local material/geometry/source gradient 回填或归约到原始 global parameter；
- interface 上的 parameter、source 和 objective 必须有有限差分测试；
- collective order 固定；不支持组合在 `prepare()` 前失败。

## 6. Phases

### Phase 0：执行契约与基准（E0, experimental）

交付：

- 冻结 `MultiGPUExecution`、`FDTDParallelConfig`、Result 顺序和错误契约；
- 建立单 GPU CUDA-event timing/memory baseline；
- 记录 `nvidia-smi topo -m`、GPU/driver/CUDA/PyTorch、dtype、grid 和 warmup；
- 完成 device pool、memory estimator 和六个 Yee component ownership 图；
- 建立单卡结果、monitor 和 checkpoint parity fixtures。

Exit gate：API review 通过；可复现 baseline；ownership 无歧义。证据等级 E0。

### Phase 1：Ensemble MVP（E2）

交付：

- ordered `run_many`；
- device lease、memory-aware placement、异常聚合和 cancellation；
- 2/4 GPU 独立 vacuum/port/sweep tasks；
- 每任务 timing、peak memory 和设备记录；
- trainable task 隔离测试。

Exit gate：结果与串行执行数值一致；任务确实并发且无 coordinator 全局同步；失败不污染其他 Result。证据等级 E2。

### Phase 2：RF-aware ensemble integration（E2）

交付：

- 01 的 `NetworkRunManifest` 把 N-port 逐激励和参数 sweep 展开为普通 Simulation tasks，并接入通用 executor；
- executor 只保证有序执行、placement、失败和 ResultSequence；01 按 public port order 组装 `Result.network: NetworkData`，02 不理解或生成 S matrix；
- 重试只针对未产生副作用的 prepare/run failure；
- 多 GPU 吞吐与单 GPU 串行对比。

Exit gate：依赖 01 Phase 2；代表性 4-port manifest 在 2/4 GPU 上返回与串行同序同值的 ResultSequence，由 01 组装的完整 matrix 同值，并展示可解释的任务级 speedup。证据等级 E2；持久化、完整失败矩阵和公开 benchmark 随 Phase 6 进入 E3。

### Phase 3：Joint-solve partition 与 P2P microbenchmark（E0, experimental）

交付：

- balanced/uneven x partition；
- component ownership、global/local mapping 和 halo metadata；
- topology preflight；
- 预分配 two-GPU CUDA P2P plane exchange；
- dedicated streams/events；
- Nsight Systems 证明 repeated exchange 无 host staging/allocation。

Exit gate：2/3/4 GPU tagged halo round-trip 正确；通信微基准报告 latency/bandwidth/break-even。证据等级 E0。

### Phase 4：标准 real-field forward MVP（E1, experimental）

交付：

- bounded E/H interior 与 boundary-strip kernels；
- eager `DistributedFDTD`；
- uniform/nonuniform local spacing；
- point source、point/time monitor ownership；
- `Simulation.prepare()` 可检查 shard layout；
- 此阶段 multi-GPU 禁用 CUDA Graph。

Exit gate：vacuum impulse/CW、uneven domain、interface source 在 1/2/3 GPU parity 达标；足够大两卡场景同时证明显存分布和正 speedup。证据等级 E1。

### Phase 5：Forward engineering preview integration（E2）

依次交付：

1. shard-aware material compiler；
2. CPML 与 PEC/PMC/Mur/symmetry global-face ownership；
3. plane/flux/mode/frequency monitors；
4. sharded full-field DFT/persistence；
5. distributed surface/plane/mode sources；
6. periodic/Bloch ring；
7. electric/magnetic ADE；
8. RF port/load runtime；
9. 03 `NetworkBlock` 与 04 circuit owner-mediated scalar reduce/scatter feature gates；
10. anisotropic/nonlinear/modulated/TFSF/SIBC feature gates。

Exit gate：默认 forward workflow 通过 L1 matrix；每 GPU persistent state 随 local volume 缩放；Result 无语义降级。证据等级 E2。

### Phase 6：Forward production hardening（E3）

交付：

- Nsight Systems 通信/计算 overlap；
- Nsight Compute 对 hot kernels 的 roofline、occupancy、register 和 memory 诊断；
- stream priority、launch bounds 和 boundary thickness 调优；
- break-even estimator，小问题明确回到单卡；
- 稳定 shard compute sequence 的 CUDA Graph（通信先留在 graph 外）。

Exit gate：命名硬件上的 scaling gates、持久化、失败矩阵、公开 benchmark 和支持矩阵全部达标，否则保持 engineering preview 并发布 profiler 证据。证据等级 E3。

### Phase 7：Joint-solve adjoint（E2）

交付：

- sharded checkpoint/replay；
- reverse halo accumulation；
- global parameter gradient reduction；
- CPML/material-density 基线；
- interface finite difference 与 repeat determinism。

Exit gate：1/2 GPU objective/gradient parity 和 interface finite difference 达标；unsupported advanced gradients 有 prepare-time guard。证据等级 E2。

### Phase 8：单节点 NCCL transport（E2）

交付：

- Linux one-process-per-GPU NCCL transport；
- timeout、rank-local logging 和 deterministic teardown；
- 与 P2P 相同 conformance suite；
- 保留 transport capability 接口，使未来 multi-node 不改变 public API。

Exit gate：单节点 P2P/NCCL 在相同 protocol 下数值一致且不改变 public API。跨节点属于 deferred future evaluation，不计入本文 target evidence 或完成定义；只有单独立项并测量 surface/volume 与网络开销后才能启动。

## 7. 验收策略

### 7.1 Ensemble correctness

- 1/2/4 GPU 输出顺序、dtype、device 和数值；
- mixed-duration tasks 无 starvation；
- prepare failure、runtime failure、OOM preflight、user cancellation；
- 单任务 gradient 不串图；
- 相同 seed 的随机工作流按既定 determinism contract 重现。

### 7.2 Joint numerical parity

比较 1/2/3 GPU：

- vacuum impulse/CW；
- source/monitor 位于 interface；
- uniform/nonuniform grid；
- 六个 field components；
- CPML、PEC/PMC/Mur/symmetry；
- periodic/Bloch；
- point/plane/flux/mode/time monitors；
- dispersion、RF ports 和逐步开放的 advanced media。

初始 FP32 gate：

- step fields：`max_abs <= 2e-6`，非近零处 `max_rel <= 2e-5`；
- monitor/DFT：`rtol <= 5e-5`，`atol <= 5e-6`；
- integrated power/flux：relative difference `<= 1e-3`；
- executed steps 一致，无 NaN/Inf。

不同 reduction order 不要求 bitwise identity；放宽阈值必须说明数值原因。

### 7.3 性能与显存

- normal step 零 H2D/D2H；
- normal step 零动态 device allocation；
- time loop 无 global `torch.cuda.synchronize()`；
- halo copy 与 interior compute 在可重叠问题上实际重叠；
- 不允许每 GPU 复制完整 field/material/DFT；
- `local state + halo overhead <= 1.15 * global state / gpu_count + measured fixed overhead`；
- 分别报告 PCIe 与 NVLink/NVSwitch；
- 对大于 break-even 四倍的问题，两卡 PCIe 必须正 speedup，NVLink weak scaling 目标 `>=70%`，peer-capable PCIe `>=50%`。

Ensemble 报告 throughput、makespan、GPU utilization 和 placement overhead；不得用 enqueue time 代替同步后的真实 wall time。

### 7.4 Benchmark family

- ensemble N-port 1/2/4 GPU；
- ensemble heterogeneous-duration sweep；
- large vacuum pulse；
- CPML dielectric scatter；
- multi-frequency full-field DFT；
- dispersive medium；
- periodic/Bloch；
- RF port/load；
- adjoint density step。

每份报告记录 global/local shape、steps、halo bytes、topology、transport、CUDA/PyTorch、graph mode、memory、数值误差和 profiler artifact。

## 8. 风险与缓解

| 风险 | 后果 | 缓解 |
|---|---|---|
| 把 ensemble 与 joint solve 混成一个布尔开关 | API 含糊且无法优化 | 两种显式 strategy，共享基础记录层 |
| Yee ownership 错误 | interface 反射或不稳定 | immutable ownership 图、single owner、interface tests |
| internal face 被当物理边界 | 假 PML/PEC | global-face flags 与 bounded kernels |
| 通信压过计算 | 小场景变慢 | break-even estimator、overlap、自动单卡选择 |
| 隐藏全局 tensor 复制 | 无容量扩展 | shard-aware compiler 与逐组件显存账本 |
| Python 调度串行化 | 多卡空闲 | profiler 驱动线程/进程/C++ coordinator 决策 |
| P2P topology 不支持 | host staging 或失败 | prepare preflight；不静默 CPU fallback |
| reduction order 漂移 | monitor/gradient 差异 | deterministic ownership/order 和量化 tolerance |
| adjoint transpose 错误 | 梯度看似合理但错误 | 明确推导 reverse exchange，interface finite difference |
| feature matrix 爆炸 | 长期不稳定分支 | 每功能单独 distributed gate，不承诺一次全开 |

## 9. 完成定义

Ensemble production complete：

- run_many 保序、失败和 cancellation 契约稳定；
- RF N-port/sweep 使用同一 executor；
- named hardware 上有真实吞吐提升；
- 没有跨任务隐式状态或梯度污染。

Joint-solve forward production complete：

- 单卡/多卡通过 L1 parity；
- 无每卡完整 state replica；
- Nsight 证明 hot loop 无 host transfer/global sync；
- 显存和 scaling gate 在命名 PCIe/NVLink 硬件上报告；
- monitor、gather、save/load 语义明确；
- unsupported combinations 在 prepare 前失败。

Adjoint 只有在 interface finite difference、checkpoint replay 和 global parameter gradient reduction 全部通过后才能声明完成。每个用户可见 level 同步更新 `FEATURE_LIST.md`、示例、benchmark 和 known limitations。
