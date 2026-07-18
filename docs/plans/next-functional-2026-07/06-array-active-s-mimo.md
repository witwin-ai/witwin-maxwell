# 阵列、Active S 与 MIMO 功能开发计划

> 状态：reopened-for-evidence（2026-07-18 审计；权重梯度成立，场景/材料梯度 fail-closed，端口功率链继承 01 风险）  
> 声明证据级：E3 production；实测证据级：**E1–E2**（96³ 资格化已于 2026-07-18 审计后 PASS，见文末追加栏）  
> 复核结论：见 `docs/assessments/next-functional-audit-2026-07-18.md` §1.4；完成记录不删除，按无通胀规则在 §14 追加"证据级实测"栏  
> 原状态（存档）：Phase 0-3 + Phase 4 权重梯度完成；场景/材料梯度 OPEN（fail-closed）；任务级多 GPU 已移除。详见 §14 的 2026-07-16 与 2026-07-18 修订  
> 日期：2026-07-14  
> 目标证据：E3 production  
> 类型：独立 RF 工程能力  
> 前置依赖：RF 基础计划交付的端口波量、`NetworkData`、accepted/available power；天线远场计划交付的有绝对功率归一化的远场结果  
> 不依赖：SPICE/MNA、thin-wire、铁氧体、SAR  
> Owner modules：拟新增 array 数据/后处理模块、`simulation.py`、`postprocess/`、`result.py`  
> 最近架构决策：2026-07-14，以功率波和 embedded element pattern 为唯一 basis，不对非线性场景使用线性叠加  
> 公共架构约束：`Scene + Simulation + Result`

## 1. 背景与当前能力

Maxwell 当前可以在一个 `Scene` 中放置多个带幅相的源，也已有 torch-native 的闭合 Huygens 面近远场、方向性/增益后处理，以及实验性的通量或模态 S 参数计算。但是这些只是组成阵列工作流的原语，并不等价于阵列产品能力。目前缺少：

- 逐端口单位入射波基函数的自动求解与可复用数据集；
- embedded element pattern（EEP）的统一功率、相位中心和极化约定；
- 基于一次基函数求解的任意复权重快速叠加；
- active reflection coefficient、active impedance、scan impedance；
- codebook、波束扫描、max-hold、EIRP 与 EIRP CDF；
- ECC、相关矩阵、diversity gain、MEG 等 MIMO 指标；
- 阵列权重和可训练场景参数的 PyTorch 梯度；
- 端口基函数任务级多 GPU 调度和域分解结果归约合同。

两份 gap analysis 都将该能力放在 RF 端口和功率链之后。原因是 active S 和 realized EIRP 不能由普通 flux ratio 或未定义功率的远场可靠推出。

## 2. 目标与非目标

### 2.1 目标

1. 以端口单位入射波为唯一基函数归一化，生成同一物理场景的完整 `S(f)` 和 EEP 数据。
2. 在不重新执行 FDTD 的情况下，对任意端口复权重计算 active S、active/scan impedance、总远场、realized gain 和 EIRP。
3. 支持单个权重、频率相关权重、批量 beam codebook 和 scan-angle 网格。
4. 提供有明确环境假设的 MIMO 指标，而不是只输出无上下文的单个“相关系数”。
5. 全部数值输出保持 `torch.Tensor`、设备和复数 dtype，允许权重优化以及后续场景伴随优化。
6. 端口基函数天然支持任务级多 GPU；域分解执行遵守统一 distributed contract。

### 2.2 非目标

- 不在本计划中实现 Lumped/Terminal/Wave Port、`NetworkData` 或 Touchstone 解析；它们来自 RF 基础与 Touchstone 计划。
- 不实现射频链路预算、调制、基带、波束赋形芯片或通信协议栈。
- 不把阵列因子近似冒充 full-wave mutual coupling；array-factor-only 必须显式标记为近似模式。
- 不默认生成统计 MIMO channel。容量等指标只有在用户提供可审计的 channel/environment model 后才计算。
- 不在第一版支持阵元位置变化后继续复用旧 EEP；几何改变会使 basis cache 失效。
- 不将离散 codebook 搜索包装成虚假的可微算子。

## 3. 用户功能描述

用户应能完成三类工作流：

1. **全波阵列基函数**：建立带 N 个 RF 端口和闭合远场面的 `Scene`，通过 `Simulation` 请求逐端口基函数，得到一个包含 `NetworkData` 与 N 个 EEP 的标准 `Result`。
2. **快速波束重组**：在 GPU 上传入 `[N]` 或 `[B, F, N]` 复权重，直接计算 B 个波束的 active S、端口功率、远场、realized gain、EIRP 和 max-hold，无需重新网格化或推进时间步。
3. **MIMO/优化**：基于全球双极化 EEP 和明确的角功率谱计算 pattern correlation、ECC、diversity gain、MEG；把权重、连续相位、场景材料或几何参数纳入 torch objective。

权重定义固定为功率波入射幅度 `a_n`，满足 `|a_n|^2` 的单位为 W。端口反射波为 `b = S a`。任何以电压、功率、相位或 dBm 给出的便利输入都必须先无损转换为该内部表示。

## 4. Public API 草案

命名在实现评审时可调整，但合同必须保持以下语义：

```python
import torch
import witwin.maxwell as mw

scene = mw.Scene(...)
scene.add_port(...)
scene.add_monitor(mw.ClosedSurfaceMonitor(name="array_nf2ff", ...))

simulation = mw.Simulation.fdtd(
    scene,
    frequencies=freqs,
    excitations=mw.PortSweep(
        ports=("p1", "p2", "p3", "p4"), amplitude=1.0,
    ),
)
result = simulation.run()

basis = result.array_basis(monitor="array_nf2ff")
weights = torch.tensor([1, 1j, -1, -1j], device=result.device) / 2
beam = basis.combine(weights)

beam.network.active_reflection
beam.network.active_impedance
beam.far_field.E_theta
beam.far_field.E_phi
beam.antenna.realized_gain
beam.antenna.eirp
```

批量 codebook 和 MIMO：

```python
codebook = mw.BeamCodebook(weights=weights_bfn, names=beam_names)
beams = result.array_basis.combine(codebook)
envelope = beams.max_hold(metric="realized_gain")

environment = mw.MultipathEnvironment(
    theta=theta,
    phi=phi,
    power_density=pas,
    cross_polar_ratio_db=6.0,
    polarization_correlation=0.0,
)
mimo = result.array_basis.mimo(environment=environment)
mimo.ecc
mimo.diversity_gain
mimo.mean_effective_gain
```

编译层新增并保持 `compile_*` 命名：

- 复用 01 的 `Scene.compile_ports(...)`、`PortSweep` 和 `NetworkRunManifest`；本文不新增第二套端口 excitation plan；
- `Scene.compile_array_monitors(...)`：解析 EEP 所需的闭合面、频率和相位参考；
- 内部 `compiler/array.py::compile_array_basis_request(...)`；
- 不新增 `Scene.set_*` / `Scene.with_*`，也不创建绕开 `Simulation` 的第二求解入口。

## 5. 数据模型与约定

### 5.1 输入模型

- `NetworkRunManifest`：直接消费 01 生成的端口顺序、每次 basis run 的 `a`、终端条件、频率集合和完成状态；array 子系统只附加 EEP/相位参考请求。
- `BeamWeights`：复数张量，尾维固定为端口维；支持 `[N]`、`[F,N]`、`[B,F,N]`，频率广播必须显式且可验证。
- `BeamCodebook`：权重、名称、目标角、状态元数据；不持有 solver。
- `MultipathEnvironment`：全球角网格、总角功率谱、极化功率比和极化相关；所有积分权重显式保存。

### 5.2 结果模型

- `ArrayBasisData`：频率、端口元数据、`NetworkData`、EEP、相位中心、球坐标基、入射功率归一化、场景/网格/cache fingerprint。
- `EmbeddedElementPatternData`：形状建议为 `[F,N,T,P]` 的 `E_theta/E_phi`，其中 `T/P` 是角维；每一列对应 `a_n=1 sqrt(W)`、其余端口匹配。
- `BeamData`：权重、`a/b`、端口 incident/reflected/accepted power、可选且带 generator provenance 的 available power、active quantities、总远场和天线指标。
- `MIMOData`：复相关矩阵、ECC、diversity gain、MEG，以及使用的 environment 元数据。

所有结果必须携带：端口参考阻抗、波定义、频率、相位参考点、角坐标约定、极化基、单位、dtype、device。序列化不得丢失复数相位或归一化元数据。

### 5.3 数学合同

- 网络叠加：`b(f) = S(f) a(f)`。
- 第 n 个端口的 active reflection：`Gamma_active,n = b_n / a_n`；`a_n=0` 时返回 mask，而不是 inf 或静默置零。
- 对实数正参考阻抗，`Z_active,n = Z0,n (1 + Gamma)/(1 - Gamma)`；复参考阻抗使用 RF 基础计划规定的 power-wave 变换，不套用此简式。
- EEP 叠加：`E_total(f,theta,phi) = sum_n a_n E_emb,n`。
- 固定使用 01 的功率恒等式：`P_incident=a^H a`、`P_reflected=b^H b`、`P_accepted=P_incident-P_reflected`；`gain=4*pi*U/P_accepted`、`realized_gain=4*pi*U/P_incident`、`EIRP=max(4*pi*U)`。available power 只在 excitation 有 generator model 时存在，不参与基础公式的任选 normalization。
- ECC 默认由全球双极化复场积分计算。只由 S 参数近似 ECC 必须使用不同方法名并标注其无损高效率假设。

## 6. 编译器、数值与运行时方案

### 6.1 Basis execution

编译器先冻结端口顺序、参考阻抗、匹配终端和 monitor 网格，再生成 N 个 one-hot excitation records。每个 record 共享同一 PreparedScene fingerprint，但拥有独立源状态、频谱累加器和结果命名空间。禁止通过一次同时激励后“猜测分离”各端口基函数。

对线性、时不变场景，N 个 basis run 可以并行执行。编译器发现 Kerr、`chi2`、时变材料、非线性电路或其他破坏叠加条件的对象时必须拒绝 basis superposition，并给出具体对象名称。

### 6.2 EEP 与远场

每个 basis run 使用相同闭合 Huygens 面和角网格。等效电流先按单位入射功率归一化，再变换为复 `E_theta/E_phi`。相位中心默认取用户指定的 `phase_center`；若未指定则用阵列几何 AABB 中心，并在结果中标记为自动值。

全波合成直接叠加复 EEP，保留 mutual coupling。另提供明确命名的 `array_factor(...)` 近似，仅用于相同孤立阵元图样和平移相位模型，结果元数据写入 `approximation="array_factor"`。

### 6.3 GPU 后处理

权重组合、`S @ a`、球面积分、max-hold 和 MIMO 相关矩阵优先用 PyTorch CUDA 张量运算。仅当 profiling 证明大量小 codebook 的 kernel launch 是瓶颈时，才增加 fused CUDA reduction；第一版不为简单矩阵运算维护一套平行原生实现。

### 6.4 Cache

`ArrayBasisData` cache key 至少包含场景物理内容、解析后网格、边界、端口编译结果、终端、频率、monitor surface、角网格、dtype、solver build id。只改变 beam weights 不使 cache 失效；任何几何、材料、端口或参考面改变都必须失效。

## 7. PyTorch/autograd 合同

- `combine(weights)` 对复权重完全可微，支持 Wirtinger 语义下的实标量 loss。
- active S、远场、增益、EIRP、ECC 和 MEG 由 torch 运算组成，不经 NumPy detach。
- 对 codebook 的 argmax/max-hold，值可提供次梯度，离散 beam index 不承诺可微；优化应使用 softmax/soft-min 等显式连续目标。
- 场景参数梯度需要对相关 basis run 分别执行 adjoint VJP，再把梯度相加；实现不得保留 N 份完整时间历史。
- basis cache 只允许复用数值值，不允许声称跨已改变参数的 autograd graph 仍有效。
- Phase 1 只承诺权重梯度；场景/几何梯度在 Phase 4 的 exit gate 后才成为公共承诺。

## 8. Multi-GPU contract

1. **任务级并行优先**：basis run 按稳定端口索引分配到设备池；每张 GPU 执行完整单 GPU 或域分解 simulation。
2. **确定性聚合**：结果始终按 public port order 重排，不按任务完成顺序拼接。
3. **域分解兼容**：单个 basis run 内由多 GPU 计划负责场 halo；闭合面样本和端口波量按 ownership 归约后才形成一列 EEP/S。
4. **驻留策略**：小型 `S` 默认复制；大型 EEP 可按 beam、频率或角块分片。`combine` 输出的分片语义必须记录，用户请求 `.to(device)` 才显式聚合。
5. **autograd**：每个 basis adjoint 使用与 forward 相同的 partition；02 executor 保持每个 task 的独立 graph，不隐式求和。06 的 array objective 在指定 result device 上按 public port order 显式组装 VJP sum，并记录 aggregation provenance。
6. **失败恢复**：每列 basis 有独立完成标记和 cache artifact，单任务失败不得损坏其他列。

## 9. Phases、交付物与 Exit Gates

Phase 0 必须冻结 `AcceptanceBudget`。默认 gate：纯 torch 解析组合 `rtol<=1e-6`；basis 与直接多源 FDTD 的复远场相对 L2 `<=0.03`、相位误差 `<=3 deg`；端口/辐射功率残差 `<=1%`；独立求解器关键 gain/ECC 指标误差 `<=0.25 dB`/`<=0.02`；权重与支持场景参数梯度相对误差 `<2%`；multi-GPU 继承 02 parity。性能必须在 Phase 0 指定硬件、任务大小和最小 parallel efficiency，不能用“接近线性”替代阈值。

### Phase 0：合同冻结与解析原型（E0, experimental）

交付物：功率波/EEP/相位中心约定；数据类草案；纯 torch `S @ a`、EEP 叠加、active S 原型；线性场景 capability guard。

依赖：RF `NetworkData` schema 已冻结。

Exit gate：2 端口解析网络和两个 Hertzian dipole 的叠加结果逐点匹配直接公式，端口/频率/角维广播和错误路径全部有测试。

### Phase 1：全波 basis 与单波束合成（E2）

交付物：复用 `PortSweep/NetworkRunManifest` 的逐端口 execution；`ArrayBasisData`；EEP；单组权重的 active reflection/impedance、total far field、realized gain、EIRP。

依赖：RF port matched termination、绝对功率远场。

Exit gate：2 阵元和 4 阵元 canonical array 在 broadside/endfire 权重下与直接多源 FDTD 的复远场、端口功率一致；basis 方法明显减少权重扫描总耗时。

### Phase 2：Codebook、扫描与任务级多 GPU（E2）

交付物：批量权重；scan grid；max-hold；beam metadata；basis cache；设备池任务调度与恢复。

Exit gate：至少 64-beam codebook 可在不重跑 FDTD 下生成；单 GPU 与 2 GPU basis 结果满足 parity；吞吐接近设备数线性扩展到通信/任务不均衡拐点。

### Phase 3：MIMO 指标（E2）

交付物：全球双极化相关矩阵、ECC、DG、MEG；环境角功率谱；可选 S-parameter ECC approximation；报告环境假设。

Exit gate：正交极化、相同图样、已知相关图样三类解析案例分别得到预期 0、1 和已知中间 ECC；相关矩阵 Hermitian positive semidefinite 在容差内。

### Phase 4：可微阵列优化与域分解接入（E3）

交付物：权重梯度；聚合 basis adjoint；连续相位/幅度约束工具；域分解 basis/EEP 归约；性能基线。

依赖：01 Phase 5 的 port/antenna adjoint；02 Phase 7 把 array basis result aggregation 加入 supported matrix。此前权重组合可微，但 scene backward 只承诺单 GPU；trainable joint solve 在 prepare 时拒绝。

Exit gate：权重梯度与 complex-step/高精度有限差分一致；至少一个材料/几何参数的 N-basis 聚合梯度通过有限差分；单 GPU与多 GPU loss/gradient parity。

## 10. 验收策略

### 10.1 单元与合同测试

- 张量 shape、广播、端口顺序、频率插值禁止项、`a_n=0` mask；
- reference impedance 与功率波转换；
- 非线性/时变 scene 拒绝；
- cache fingerprint 与序列化 round-trip；
- 复权重梯度和 dtype/device 保持。

### 10.2 解析验收

- 各向同性点源阵列因子：2、4、8 阵元 broadside/endfire；
- 均匀线阵主瓣方向、零点、半功率波束宽度；
- 已知 `S` 矩阵下的 active Gamma/impedance；
- 正交与相同极化图样的 ECC 极限。

### 10.3 独立求解器/测量验收

- 2/4 元 patch 或 dipole 阵列与独立商业求解器或公开测量数据比较 S、EEP、scan impedance 和 realized gain；
- 结果 artifact 固定 solver 版本、网格、端口参考面和原始复数数据，不能只保存图。

### 10.4 收敛、守恒与互易性

- 空间网格、时间长度、角积分网格三重收敛；
- 对无源阵列检查入射功率 = 反射 + 辐射 + 材料/端口耗散；
- 互易无源结构检查 `S_ij≈S_ji`，但不把互易性强加给铁氧体等非互易场景；
- EEP 叠加辐射功率与端口网络接受功率的差额在既定容差内。

### 10.5 性能验收

- 记录 basis solve、NF2FF、codebook combine 的时间和峰值显存；
- 64/256/1024 beams 的后处理不得触发 FDTD；
- 任务级 2/4 GPU strong scaling；
- EEP 分块后处理的峰值显存随 block size 而非 beam 总数增长。

## 11. Benchmark 矩阵

| 场景 | 主要指标 | 参考 |
| --- | --- | --- |
| 2 个 Hertzian dipole | 复远场、阵列因子、功率 | 解析 |
| 8 元均匀线阵 | 扫描角、零点、HPBW | 解析 array factor |
| 2 元耦合 dipole | `S`、active impedance、EEP | NEC/MoM |
| 4 元 patch 阵列 | `S`、scan impedance、realized gain | 独立求解器/公开测量 |
| 双极化 2-port 天线 | ECC、DG、MEG | 解析极限 + 独立数据 |
| 64-beam codebook | max-hold、EIRP CDF、吞吐 | CPU/torch reference + 性能基线 |

benchmark 场景放入 `benchmark/scenes/array/`，结果汇总进 `benchmark/RESULTS.md`；任何阈值变化必须保留原因和前后数据。

## 12. 风险与缓解

- **错误归一化产生看似合理的图**：basis、EEP、gain 全链路只使用一种功率波定义，并以能量守恒 gate 拦截。
- **相位中心不一致**：相位中心属于持久化数据合同，合成前强制一致。
- **非线性场景误用叠加**：编译期 capability guard，不提供 silent fallback。
- **EEP 数据过大**：频率/角块分片、lazy combine、可选球谐压缩；压缩必须有误差界。
- **MIMO 指标被环境假设支配**：结果始终携带 environment，未提供环境时不输出 MEG/统计容量。
- **N 次 adjoint 成本高**：只对 objective 涉及的 basis 运行 VJP，聚合 seed，并支持任务级多 GPU。

## 13. 完成定义

本计划完成必须同时满足：

1. 标准入口仍为 `Scene -> Simulation -> Result`，可生成可序列化 `ArrayBasisData`。
2. 任意批量权重在 GPU 上得到 active S、active impedance、复远场、realized gain 和 EIRP，不重新运行 FDTD。
3. ECC/DG/MEG 具有明确全球双极化与环境合同。
4. 解析、独立求解器、收敛、守恒、互易性和性能 benchmark 全部通过。
5. 权重 autograd、至少一个场景参数的 basis-adjoint，以及单/多 GPU value/gradient parity 通过。
6. 用户可见能力同步更新 `FEATURE_LIST.md`，示例只使用公共 API，无直接 backend 构造。

## 14. 修订记录（append-only，不重写历史）

### 2026-07-16 修订

以下修订对上文正文（尤其 §2.1 目标 6、§8、§9 Phase 0/Phase 2/Phase 4）作出增补与更正。上文正文保留原样以存档决策历史；如与本节冲突，以本节为准。

#### (a) 移除任务级多 GPU 范围

经用户 2026-07-16 明确决定，从本计划范围中移除任务级多 GPU（device-pool 任务调度、1/2/4-GPU strong-scaling gate 及相关 parallel-efficiency 阈值）。相应地：

- §2.1 目标 6、§8 全节、§9 Phase 2 的“设备池任务调度与恢复 / 2 GPU parity / 吞吐接近设备数线性扩展”、§9 Phase 4 的“单 GPU与多 GPU loss/gradient parity”不再作为本计划的交付承诺；
- `AcceptanceBudget` 不再携带 `two_gpu_parallel_efficiency`、`four_gpu_parallel_efficiency`、`scaling_hardware`、`task_s_rtol`、`task_s_atol`、`task_basis_count` 等字段，其缺席即合同（由 `tests/rf/array/test_array_contracts.py::test_acceptance_budget_carries_no_cancelled_task_level_multi_gpu_scope` 锁定）。

域分解（单个 basis run 内的多 GPU 场求解）不属于本次移除范围，仍由 02 计划的 distributed contract 承接。

#### (b) 收紧 basis-vs-direct 门限至本分支冻结值

§9 Phase 0 原文规定 basis 与直接多源 FDTD 的复远场相对 L2 `<=0.03`、相位误差 `<=3 deg`、端口/辐射功率残差 `<=1%`。这些工程级容差被证明对本 gate 没有判别力：basis-vs-direct 比较是同一求解器上的线性叠加自洽性检查，必须绑定到求解器精度而非工程容差；`0.03`/`3 deg`/`1%` 无法把真实的叠加回归与数值噪声区分开。据此收紧为下列冻结值（由 `witwin/maxwell/array.py::AcceptanceBudget` 定义、`test_phase_zero_acceptance_budget_is_frozen` 锁定），两个运行该 gate 的场景因截断误差不同而各持一套阈值：

| 门限字段 | 旧值 | 新冻结值 | 实测最坏值（证据） |
| --- | ---: | ---: | --- |
| `contract_fdtd_complex_l2`（粗合同场景） | 0.03 | 5.0e-3 | 1.433e-4（四元 endfire，4 层 PML） |
| `contract_fdtd_phase_rms_deg`（粗合同场景） | 3.0 deg | 0.5 deg | 6.776e-3 deg（四元 endfire，4 层 PML） |
| `phase1_fdtd_complex_l2`（Phase 1 收敛基准场景） | 0.03 | 1.0e-4 | 2.219e-6（endfire） |
| `phase1_fdtd_phase_rms_deg`（Phase 1 收敛基准场景） | 3.0 deg | 1.0e-2 deg | 1.518e-4 deg（endfire） |
| `port_power_relative_error` | 0.01（1%） | 5.0e-3 | 9.015e-6（两元 endfire，4 层 PML） |
| `radiated_power_psd_relative_floor` | -1e-3 相对下限 | 1e-9 相对下限（PSD + `max_eig>0`） | 见 (c) |

阈值再次变更须保留旧值、实测证据与技术理由（沿用 §11 的变更规则）。

#### (c) 粗合同测试场景改用 4 层 PML

`tests/rf/array/test_array_fullwave.py` 的粗合同场景由 2 层 PML 提升为 4 层 PML。根因分析：`Q_rad` 是闭合面复 Poynting 算子的 Hermitian（实功率）部分，理论上正半定，但 2 层 PML 欠吸收——`0.05 m` 的 NF2FF 盒距 `0.06 m` 内域边界仅约 1 个网格，反射场污染闭合面 Poynting 积分，使最小特征值随 PML 深度变号（2 层：四元 min/max = -2.833e-5；4 层：四元 min/max = +1.449e-6，两元 = +4.215e-2）。此前用 `-1e-3` 相对下限掩盖了该欠吸收伪迹；改用 4 层 PML 后谱真正正定，门限只保留 `eigvalsh` 浮点舍入带（`radiated_power_psd_relative_floor = 1e-9`）并附加 `max_eig>0`，`test_array_fullwave.py` 与 benchmark 两处同时执行，杜绝整体变号仍通过。

### 2026-07-18 完成记录（Phase 0-3 + Phase 4 权重梯度完成，master `3f13ff2`）

本节追加 Phase 完成状态与两条晚于本分支的 convention 更正，正文与前述修订保留原样；如冲突以本节为准。

#### (a) Phase 0-3 完成；Phase 4 权重梯度完成

Phase 0-3 已完成并落地。Phase 4 的**权重梯度**部分完成，验收记录为 `docs/assessments/array-active-s-mimo-phase-4-acceptance.md`（Status: weight-gradient gates accepted）：

- 复数入射功率波权重梯度贯穿 `ArrayBasisData.combine`、批量 codebook 路径、`BeamData.max_hold`（子梯度值 / 非可微获胜 beam index）与 MIMO 指标 kernel，**零额外 FDTD step**；证据 `tests/rf/array/test_array_contracts.py::test_complex_weight_gradient_matches_high_precision_gradcheck` 及 `test_array_codebook.py` 的 gradcheck leg（complex128, eps 1e-6, atol 1e-5, rtol 0.02）。
- MIMO vs 独立 Clarke 闭式：`test_array_mimo.py::test_polarization_correlation_cross_term_matches_brute_force_integral`（`rho != 0` 双极化交叉项对照独立梯形积分）；PSD `Q_rad` 见上 (b)/(c) 冻结值与 §5.3。

#### (b) OPEN 项（已记录，fail-closed）

- **场景/材料/几何梯度贯穿 basis**：sliced out 且 **fail-closed**——`ArrayBasisData.scene_gradient_vjp(...)` 抛 `NotImplementedError`（不返回静默 `None`），作为一条新 capability guard 结算（census 132→133，`tests/api/public/test_guard_census.py`；disposition 见 `docs/reference/fdtd-capability-guard-census.md`）。成为公共承诺需待聚合的 per-column adjoint 落地，依赖 02 Phase 7 分布式 result-aggregation contract。
- 冻结的 `96^3 / 4096-step` 资格化运行 + Phase 1 门限再资格化，需在**独占计时窗口**执行，仍 deferred。
- 任务级多 GPU：按 §14 的 2026-07-16 用户决定，保持 **REMOVED**（不再在本计划范围内）。

#### (c) 两条晚于本分支的 convention 更正

以下两条更正在本分支之后落地，纠正了本分支引入的错误约定：

1. **observer DFT 约定改回 PLAIN step phase（E 与 H）。** 本分支曾给 running-DFT observer 加入 E/H stagger twiddle（`field_offset`），导致点/面 spectral observer 相对全场 DFT 累加器出现约 21% 均匀相位误差，破坏 spectral-observer、field-time-monitor、dipole-emission 与 dielectric half-space S11 检查。该 twiddle 及其 adjoint transpose 已被回退，E 与 H 均以 plain running-DFT step phase 累加（master commit `21be130`，pinned by `tests/fdtd/test_observer_quadrature.py` 及 observer time-stagger 回归）。
2. **NF2FF cell-centered quadrature 裁剪到采样的 Huygens 面。** 原等效电流辐射积分对每个 cell-centered 样本按整格 primal 宽度加权，使每个有限面向外多算半格并重复计边界 cell，over-count 远场（闭合面辐射功率过一、Rayleigh RCS 偏高、介质半空间反射参考偏强）。修正把边界控制宽度裁到最外层样本（均匀网格上的梯形边界规则），内部保留精确 primal 宽度（master commit `3f13ff2`，pinned by `tests/fdtd/test_observer_quadrature.py`；`witwin/maxwell/postprocess/stratton_chu.py`）。

### 2026-07-18 证据级实测（审计回退，不删除上文完成记录）

本栏依 `docs/assessments/next-functional-audit-2026-07-18.md` §1.4 与 §4 的无通胀规则追加。上文修订与完成记录逐条保留存档；本栏只登记**实测证据级、欠账，以及一条审计后的正向更新**，如与上文的完成声明冲突，以本栏对证据级的判定为准。

- **实测证据级：E1–E2**（非声明的 E3）。Phase 0-3 + Phase 4 权重梯度成立：codebook/max-hold/MIMO/ECC 全在 autograd 图内、gradcheck 通过，MIMO vs 独立 Clarke 闭式与 PSD `Q_rad` 正定（4 层 PML）为可信 E2 级证据。
- **场景/材料/几何梯度 fail-closed**：`ArrayBasisData.scene_gradient_vjp(...)` 抛 `NotImplementedError`（census 132→133，不静默返回 `None`），依赖 02 Phase 7 分布式 result-aggregation contract 方能成为公共承诺。
- **继承 01 端口功率链风险**：全部 EIRP/realized gain 继承 `01` 的 accepted/available power 约定，而该约定未经 wave 级验证（审计 §1.1）。
- **审计后正向更新（2026-07-18，post-audit）**：冻结的 `96^3 / 4096-step` 资格化已在本 host（2x RTX A6000，独占窗口）执行并 **PASS**。物理功率闭合 `|P_accepted - P_rad| / P_incident = 6.971e-4（0.0697%）`，远在 1% 冻结门内；timing contract 达标（basis+16-combine 为 16 次直接求解的 20.55%，门 <= 40%；单 combine 为单次直接求解的 0.0343%，门 < 10%，零额外 FDTD step）；`local_hardware` 已重锚到本 host。**工件：`docs/assessments/array-active-s-mimo-phase-1-qualification.json`（`verdict=PASS`，commit `1cc4a71` 修正后的 flux/observer 约定）。** 该资格化即上文所列"deferred-pending-exclusive-window"项的结算，审计数据（2026-07-18 之前）未包含它。
- **提升到 E2/E3 所需证据（收敛路线，见审计 S3.3）**：
  1. 完成 `scene_gradient_vjp` 聚合 per-column 伴随、关闭 fail-closed（依赖 02 Phase 7）；
  2. 阵列因子解析 + Tidy3D（🟡，需能力落地确认，否则标 `future-xfdtd`）多场景对照进入 RESULTS；
  3. 端口功率链在 01 完成 S1 wave 级验证后再继承（提 E3 的组合矩阵/公开 benchmark，README §7 定义）。
- 进入门：本计划 S3.3 收敛工作阻塞于 S1（01 端口 wave 级验证）先行通过。
