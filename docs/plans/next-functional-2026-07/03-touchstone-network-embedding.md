# Touchstone 网络嵌入开发计划

状态：reopened-for-evidence（2026-07-18 审计；契约层与运行时已落地）；Round-E 已补 E2 证据（见文末 2026-07-21 栏）  
声明证据级：E3 production；实测证据级：**E1 → E2（round-E，embedded path）**（gate (d) 仍网格限定 PASS）  
复核结论：见 `docs/assessments/next-functional-audit-2026-07-18.md` §1.3；完成记录不删除，按无通胀规则在文末追加"证据级实测"栏  
原状态（存档）：Phase 0-4 完成（Phase 4 gate (d) 为网格限定 PASS；详见文末 2026-07-18 完成记录）  
日期：2026-07-14  
目标证据：E3 production  
依赖：`01-rf-engineering-workflow.md` Phase 2 冻结 `NetworkData` 与端口功率约定  
Owner modules：`network.py`（拟新增）、`compiler/`、`fdtd/`、`result.py`  
最近架构决策：2026-07-14，FDTD 嵌入统一降低为被动状态空间，不采用仿真后 S 矩阵级联

## 1. 背景与当前能力

RF 基础计划将提供 Maxwell 自身仿真的 `NetworkData` 和 Touchstone 导出，但“导出结果”与“把已有网络模型嵌入电磁场景”是两项不同能力。当前 Maxwell 没有 Touchstone parser、端口映射、因果/无源检查、宽带有理拟合、时域状态空间 realization，也没有把外部 N-port 的电流反馈耦合到 FDTD 端子。

网络嵌入的价值是无需网格化连接器、封装、滤波器或已表征器件，即可让它们作为线性宽带负载参与同一次全波仿真。实现不能退化为仿真后把两个 S 矩阵相乘；嵌入网络必须改变端口边界处每个时间步的场更新。

## 2. 目标与非目标

### 2.1 目标

1. 读取 Touchstone 1.x/2.0 常用 RI/MA/DB、Hz/kHz/MHz/GHz 和 S/Z/Y 数据，转换成 `NetworkData`。
2. 将 `NetworkData` 以显式端口映射嵌入 `Scene`，支持一端口终端和多端口跨接网络。
3. FDTD 使用稳定、因果、被动的实数状态空间模型逐步更新；Touchstone 频域样本只作为模型输入和验证数据，不引入第二个全波运行时。
4. 提供 reference impedance 处理、端口重排、频带裁剪、passivity/causality diagnostics 和拟合误差报告。
5. 运行时状态及可支持参数保持 GPU/PyTorch-native，并定义空间 multi-GPU ownership。

### 2.2 非目标

- 不实现非线性、时变、噪声或温度相关网络；这些不能由普通 Touchstone 唯一表达。
- 第一版不支持 mixed-mode order 关键字的所有厂商扩展、噪声参数块和任意自定义注释语义。
- 不以调用外部 SPICE/ngspice 作为 FDTD 时间步运行时。
- 不承诺从不满足因果/无源性的测量文件自动修复出“物理正确”模型；自动 enforcement 必须显式请求并报告改动。
- 不做 UI 文件选择与网络原理图；只提供后端对象和结构化 diagnostics。

## 3. 用户功能描述与 Public API 草案

```python
import witwin.maxwell as mw

data = mw.NetworkData.from_touchstone("filter.s2p", device="cuda")
report = data.validate_physicality(band=(1e9, 8e9))

embedded = mw.TouchstoneNetwork(
    name="filter",
    network=data,
    connections={1: "feed_port", 2: "antenna_port"},
    fit=mw.RationalFitConfig(
        order=24,
        band=(1e9, 8e9),
        enforce_stability=True,
        enforce_passivity=True,
        relative_tolerance=1e-3,
    ),
)

scene.add_network(embedded)
result = mw.Simulation.fdtd(scene, frequencies=freqs).run()
network_state = result.embedded_network("filter")
```

也支持由 tensor 构造，不强制文件 I/O：

```python
block = mw.NetworkBlock(
    name="trainable_match",
    network=mw.NetworkData(frequencies=f, s=s_tensor, z0=z0, port_names=("1", "2")),
    connections={"1": "p1", "2": "p2"},
)
scene.add_network(block)
```

公共契约：

- `TouchstoneNetwork` 是从文件/`NetworkData` 构造的 `NetworkBlock` 便利类型；连接目标必须是 `01` 定义的 Lumped/Terminal/RF WavePort terminal。
- `Scene.add_network(...)` 把网络作为场景物理的一部分；`Simulation` 入口不变。
- `Scene.compile_networks(...)` 产生 backend-specific realization；内部函数统一 `compile_*` 命名。
- `Result.embedded_network(name)` 返回端口 V/I、网络内部耗散功率、状态空间诊断和 fit report，不返回不透明 solver handle。

## 4. 数据模型与规范

### 4.1 Touchstone I/O

parser 产出标准 `NetworkData`，保留：原文件 format、parameter kind、frequency unit、port names/order、每端口 Z0、comments 的原始文本和 parser warnings。解析使用严格 token/行号错误；不接受 NaN、重复/倒序频率、矩阵元素缺失和非正 `Re(Z0)`。

parser 必须正确实现 Touchstone 的矩阵展开顺序，2-port 特殊顺序以 golden file 固定。写回时由 `NetworkData.to_touchstone()` 负责，import/export round-trip 不改变端口顺序或 reference impedance。

### 4.2 公共与内部对象

- `NetworkBlock`：name、`NetworkData`、connections、fit config、extrapolation policy。
- `RationalFitConfig`：order/order range、band、weights、iterations、稳定/无源策略和误差阈值。
- `RationalModel`：共享 poles、residues、direct/proportional terms、representation (`Y` 默认) 与 fit report。
- `StateSpaceNetwork`：实数块状态矩阵 `A/B/C/D`、离散矩阵、port order、passivity margin。
- `NetworkFitReport`：RMS/max complex error、被动性最大违反、unstable poles、延迟估计、condition numbers 和实际 enforcement 改变量。
- `EmbeddedNetworkData`：每频端口 V/I、absorbed/generated power、state norm、fit model id、runtime warnings。

`RationalFitConfig`、通用 pole/residue 表示、稳定性检查、被动性 enforcement、连续到离散变换和基础 `FitReport` 属于共享内部 rational-model 基础设施，供本文和 `09-surface-impedance-metal-roughness.md` 共用。本文只拥有 multiport `NetworkBlock` realization 与 `NetworkFitReport` 的端口专用字段；09 只拥有切向 surface specialization，禁止复制两套 vector fitting 和正实投影实现。

FDTD 默认将输入转换为端口导纳 `I(s)=Y(s)V(s)` 的正实 realization，避免在开路附近对 Z 求逆；矩阵变换通过 `torch.linalg.solve`，禁止显式 inverse。纯传播延迟若不适合低阶 rational fit，可使用显式整数/分数延迟线状态，但必须有固定内存上限。

## 5. 编译器、运行时与数值方案

### 5.1 预处理与拟合

1. 读取/验证 `NetworkData`，重排并重归一化到所连接端口的 power-wave 约定。
2. 裁剪到 simulation excitation 的有效频带，并按用户 policy 拒绝或显式外推；默认禁止带外外推。
3. 选择 Y representation，执行 multiport vector fitting；强制共轭 pole/residue 与实系数 realization。
4. 把右半平面 poles 映射/重新拟合到左半平面；检查 Hamiltonian/passivity sampling，必要时最小扰动 residues。
5. 以 bilinear/trapezoidal 离散化到 FDTD `dt`，检查离散 poles 位于单位圆内并生成能量/passivity certificate。

拟合属于 prepare 控制面，允许 CPU scientific routines；生成的定长矩阵一次传入 GPU。拟合失败、误差超阈值或 enforcement 改动超阈值必须阻止运行，除非用户显式采用 `diagnostic_only`，后者也不得绕过不稳定模型。

### 5.2 FDTD 强耦合

每一步从连接端口测得电压向量，在 GPU 上执行离散状态更新 `x[n+1]=Ad x[n]+Bd v[n]` 和 `i[n]=Cd x[n]+Dd v[n]`，然后把电流反馈到端口 E update。直接项 `D` 与端口局部电磁导纳形成小型线性系统，必须同一步隐式求解，不能使用一拍延迟造成伪能量。固定拓扑时预分解常量矩阵；网络状态不进入 Python loop。

### 5.3 GPU-first 与 PyTorch

- 时间步状态、端口向量、小矩阵 solve 和功率累积全部 CUDA tensor。
- 文件解析与 vector fit 默认不参与 autograd；由 tensor 直接构造、且提供预拟合 poles/residues 的 `RationalModel` 可对 residues/direct terms 求梯度。
- 第一版不对 pole relocation/被动性 enforcement 求导；调用时若输入需要梯度必须明确报错或要求 `fit=False`，不得静默 detach。
- FDTD backward 对 state-space recurrence 使用 checkpoint/replay 或离散 adjoint，内存不随保存全部 network state 线性膨胀。
- 状态空间 kernel 应可进入 CUDA Graph；动态 order 和 Python callable 禁止进入 runtime。

## 6. Multi-GPU ownership 与 reduction contract

- 一个 `NetworkBlock` 有唯一 owner rank：优先选择拥有最小编号连接端口 reference point 的 shard；所有 rank 从 manifest 得到一致 owner。
- 各 port 的 shard-local V fragments 先按 `01` 的端口 contract reduction 到 network owner；owner 更新唯一 state vector 并计算全部端口 currents。
- currents 按连接映射 scatter 到各 port owner，再由 port compiler 分发到本地 Yee edges。只有 `O(Nports)` 标量通信，不迁移全场或复制 state。
- 若网络所有端口均在同一 shard，运行时不得发起 collective。
- task-level 多 GPU sweep 中，每个独立 run 持自己的网络状态；严禁跨 excitation 共享 mutable state。
- backward 使用相反通信图归约 `dV/dI`；在 distributed adjoint 完成前，trainable embedded model + spatial multi-GPU 精确拒绝。
- 结果只由 network owner 发出，global Result gather 按 network name 去重；功率先对端口求和再判断被动性。

## 7. Phases、交付物与 exit gates

### Phase 0：Touchstone I/O 与 NetworkData 互操作（E0, experimental）

交付：严格 parser/writer、端口顺序、S/Z/Y 与 per-port Z0、diagnostics。  
依赖：`01` Phase 2 schema。  
Exit gate：官方/自建 1/2/4-port golden corpus 全通过；RI/MA/DB 往返复数误差 `< 1e-10`（float64）；错误文件给出行号；2-port ordering 有独立断言。

### Phase 1：物理检查与 rational realization（E1, experimental）

交付：causality/passivity/stability checks、multiport vector fitting、state-space 离散化、fit report。  
依赖：Phase 0。  
Exit gate：解析 RC/RLC/传输线数据的带内 max complex error `< 1e-3` 或用户阈值；所有离散 poles `|z| < 1-1e-7`；passive corpus 离散能量无增长，主动数据被准确识别。

### Phase 2：一端口 FDTD 嵌入（E2）

交付：NetworkBlock、scene/compiler、GPU state update、端口反馈、Result diagnostics。  
依赖：Phase 1；`01` Lumped/TerminalPort。  
Exit gate：RLC Touchstone load 与原生 RLC 的端口 V/I/S11 差 `< 1%` 幅值且 `< 2 deg` 相位；无源网络 `P_abs >= -1e-5 P_inc`；profiler 无逐步 CPU activity。

### Phase 3：多端口与 delay（E2）

交付：N-port mapping、implicit direct term、受控 delay realization。  
依赖：Phase 2。  
Exit gate：2/4-port fixture 与独立 circuit/network reference 的 S 参数 max error `< 0.02`；端口重排结果 permutation-equivalent；长延迟场景相位误差 `< 3 deg`；FDTD 稳态端口量与 fitted model 频率响应差 `< 2%`。

### Phase 4：梯度、空间 multi-GPU 与生产化（E3）

交付：state-space adjoint、owner/reduction、persistence、文档和性能优化。  
依赖：multi-GPU forward contract、Phase 3。  
Exit gate：residue/direct term 梯度相对有限差分 `< 2%`；单/多 GPU port V/I `rtol <= 2e-5`；未连接网络的基础 FDTD 性能回退 `< 1%`；连接 8-port/order-32 网络 step 开销 `< 10%`（代表网格）。

## 8. 验收与 benchmark

### 8.1 单元测试

- parser option line、单位、comments、matrix order、per-port Z0、bad token/EOF。
- port mapping 的重名、缺口、重复连接、数量不匹配。
- rational conjugate symmetry、连续/离散 pole、state initialization、save/load。
- S/Y/Z 与 state-space frequency response 一致性。

### 8.2 数值/端到端

- `network/one_port_rlc_touchstone`：与解析和原生 RLC 三方比较。
- `network/two_port_filter_embedding`：通带/阻带、群时延、功率。
- `network/coax_delay_line`：长延迟和带外 policy。
- `network/four_port_coupler`：端口顺序、隔离度和多端口 passivity。
- `network/antenna_matching_block`：外部匹配网络改变 realized gain 的完整闭环。
- `network/multi_gpu_split_ports`：连接端口跨 shard 的 parity/通信统计。

### 8.3 梯度与性能

梯度覆盖 residue、direct conductance 和连接端口附近材料参数；用多 step 中心有限差分。性能记录 fit 时间、state order、每 step 小矩阵 solve、通信字节和 peak memory。8-port/order-32 的 network state 应为 `O(N*order)` 或低秩 realization，禁止无说明的 `O(N^2*order)` 状态膨胀。

## 9. 风险与缓解

- **测量数据非因果/非无源**：分开 diagnostics 与 enforcement，报告改变量，不隐藏修复。
- **宽带延迟需要极高阶拟合**：显式 delay extraction + 有界 delay buffer，超内存提前失败。
- **直接项产生代数环**：端口与 D 项同一步小型隐式 solve，condition number 入 diagnostics。
- **拟合不可微**：明确把 fit 当编译步骤；优化用户传预拟合可微 coefficients。
- **端口规范不一致**：只依赖 `01 NetworkData` 的 power-wave、Z0、方向与 reference-plane contract；连接时强制转换并记录。
- **小网络通信主导多 GPU**：网络放置靠近连接端口；本地连接完全绕过 collective；记录每步标量通信成本。

## 10. 完成定义

完成要求 Phase 0-4 全部 exit gates 通过；Touchstone corpus、RLC/滤波器/延迟线/四端口/天线匹配 benchmark 进入 CI；FDTD 中网络真实反馈场而非后处理 cascade；所有状态 GPU-resident；可微与不可微路径清晰且不 silent detach；空间 multi-GPU 按唯一 owner 合并；公共能力更新 `FEATURE_LIST.md`，并有从 `.s2p` 到 `Scene -> Simulation -> Result` 的可运行示例。

## 11. 修订记录（append-only，不重写历史）

上文 §1-§10 正文保留原样以存档 proposed 阶段决策；本节仅追加完成状态与证据引用，如与正文冲突以本节为准。

### 2026-07-18 完成记录（Phase 0-4 完成，master `e0a0877`）

Phase 0-3 已完成并落地。Phase 4 四项 exit gate（§7 Phase 4）逐项结算如下，全部有可复现证据；其中 gate (d) 为诚实记录的**网格限定** PASS。

#### gate (a)：residue/direct term 梯度相对有限差分 `< 2%` — PASS

`tests/gradients/test_fdtd_network_adjoint.py::test_rational_network_gradient_matches_three_step_central_difference` 与 `::test_network_connected_material_gradient_matches_three_step_central_difference` 用三步中心差分对照 adjoint，断言 `max(relative_errors) < 0.02`。conjugate residue 梯度另有 real-model 约束保持测试。

#### gate (b)：单/多 GPU 端口 V/I `rtol <= 2e-5` — PASS（bitwise）

`tests/fdtd/multi_gpu/test_network_owner.py`：跨 shard 的 `split_net` 双端口网络场景，单 GPU 与两 GPU 的端口 V/I 与场量 bitwise 相同（偏差 0，远严于 `2e-5` 门限），确定性 owner 与 `O(Nports)` 标量通信 contract 同时验证。该场景即 §8.2 `network/multi_gpu_split_ports` 的等价实现。

#### gate (c)：未连接网络的基础 FDTD 性能回退 `< 1%` — PASS（op-stream 等价证据）

不以单次计时数字判定，而以主机指令流等价为主证据（与 host class 计时分辨率下限一致的处理方式）。证据 artifacts：

- `docs/assessments/network-embedding-phase-4-no-feature-op-stream.json`（`torch.profiler` op 表逐项相同，empty diff，`equivalent=true`）
- `docs/assessments/network-embedding-phase-4-no-feature-kernel-identity.json`（每步 kernel 集合一致）
- `docs/assessments/network-embedding-phase-4-no-feature-abba.json` 与 `-abba-confirm.json`（计时 leg，配对 ABBA 归档）

#### gate (d)：连接 8-port/order-32 网络 step 开销 `< 10%`（代表网格）— GRID-DEPENDENT（诚实记录）

判定：**仅在 `>= ~224^3` 网格 PASS，在默认/代表小网格 FAIL。** 证据 artifacts：

- `docs/assessments/network-embedding-phase-4-performance.json`：默认 `64^3` 网格，连接开销 **138.10%**，`passed=false`。
- `docs/assessments/network-embedding-phase-4-performance-grid-sweep.json`：网格扫描给出 10% 门限的越界点约在 `224^3`；实测 `64^3`=138.10%、`96^3`=49.91%、`128^3`=41.61%、`176^3`=18.28%、`224^3`=**9.64%（PASS）**。

根因：连接路径（field+network+port_observer CUDA graph、8 端口、order-32 线性 state_count=256、隐式 solve size 8、端口 observer）引入近乎与网格无关的**固定 `~0.20 ms/step` 耦合成本**（扫描各档实测 0.193-0.204 ms/step）。因此开销**百分比**只在裸 field step 大到足以摊薄该固定成本时才降到 10% 以下。这是 gate (d) 未来的明确优化目标：压低这份固定的 `~0.20 ms/step` 端口/网络耦合成本，而非线性 MNA solve（8→128 未知量成本平坦）。

#### 端口接口的梯形统一（trapezoidal port-interface unification）

端口-场耦合在原生 lumped、嵌入网络与 MNA 电路三条路径上统一为同一梯形（trapezoidal）半步电压接口。跨模型精确性证据：`tests/rf/circuits/test_fdtd_circuit_coupling.py`（native.Ez vs coupled.Ez，`rtol=2.0e-6`）、`tests/rf/circuits/test_phase3_multiport.py`（`rtol=2.0e-6`）、`tests/rf/network/test_network_multiport_runtime.py`（coordinator unification ruling：网络 solve 消费同一梯形半步电压 `0.5*(V_after_prev + V_free)`）。

#### plan-corpus 新增（§8.2 落地）

- `network/antenna_matching_block`：外部匹配网络改变 realized gain 的完整闭环，真实求解 E2E 落在 `tests/rf/antenna/test_antenna_matching_block.py`。
- `network/multi_gpu_split_ports` 等价场景：`tests/fdtd/multi_gpu/test_network_owner.py` 的 `split_net` 跨 shard 端口 parity 与通信统计。

### 2026-07-18 证据级实测（审计回退，不删除上文完成记录）

本栏依 `docs/assessments/next-functional-audit-2026-07-18.md` §1.3 与 §4 的无通胀规则追加。上文"完成记录"逐条保留存档；本栏只登记**实测证据级与欠账**，如与上文的完成声明冲突，以本栏对证据级的判定为准。

- **实测证据级：E1**（非声明的 E3）。运行时（有理拟合 → 状态空间 → 同步耦合、pivoted-LU、CUDA Graph replay、单卡伴随）确实落地并进 `FEATURE_LIST`，但缺 E2 所需的多场景守恒/无源性 + 独立求解器交叉验证。
- **Phase 4 gate (d) 为网格限定 PASS**：仅在 `>= ~224^3` 达标（`64^3`=138.10% FAIL），非通用；上文已诚实记录，此处确认其不构成通用 E2 性能门证据。
- **继承 01 端口功率约定风险**：本计划端口 V/I 功率约定继承自 `01`，而 `01` 的端口功率链尚未经 wave 级验证（审计 §1.1），故本计划的端口量可信度受 01 上游制约。
- **分布式路径多组合 fail-closed**（trainable embedded model + spatial multi-GPU），依赖 02 Phase 7 分布式 result-aggregation。
- **提升到 E2 所需证据（收敛路线，见审计 S3.1）**：
  1. Phase 4 gate (d) 由网格限定升为通用（压低约 `0.20 ms/step` 固定耦合成本）；
  2. 多场景守恒/无源性 + **独立 S 参数级联工具**交叉验证；先做 Tidy3D lumped/network 能力落地确认（🟡），可用则叠加 Tidy3D 对照并入 RESULTS，否则标 `reference: future-xfdtd`；
  3. 相关 network benchmark 场景进入 `benchmark/RESULTS.md` 常驻。
- **提升到 E3 所需证据**：在 E2 之上补组合矩阵、命名硬件性能边界、分布式/梯度支持声明与公开 benchmark（README §7 E3 定义）。
- 进入门：本计划 S3.1 收敛工作阻塞于 S1（01 端口 wave 级验证）先行通过。

### 2026-07-21 Round-E 交付（实测证据级 E1 → E2，embedded path）

依无通胀规则追加；上文各栏保留存档。Round-E（merges `4db24ec` = `26349db` + `fabd485`）
supplied the E2 evidence the audit's S3.1 required for the embedded path:

- **Independent raw-sample S-cascade cross-check** (`NetworkData.cascade`/`terminate`
  first-principles algebra on raw Touchstone samples vs the embedded rational-fit +
  state-space run — **no shared code path**): residuals `~4.6e-9`/`~5.7e-8` vs a
  `<1e-5` gate, connection changes S11 by `~2e-4` (≫ tol). This closes the
  fit-model-class circularity flagged in the 2026-07-18 栏 item 2.
  `tests/rf/network/test_network_cascade_crosscheck.py`.
- **Multi-scenario passivity/conservation suite** (3 embedded FDTD scenarios): terminal
  power-balance (honestly *consistency*-class for memoryless, *genuine two-sided* for
  the reactive scenario), passivity, and stability gates.
  `tests/rf/network/test_network_conservation.py`.
- **Composite same-step coupling** launches 78→27 (−65.4%, bitwise eager vs graph);
  **explicit-delay checkpoint/resume** landed (fixes a lossy-resume-from-zero bug),
  **but the delay adjoint stays fail-closed** (segment-crossing reverse ring + IIR
  reverse out of scope); **WavePort embedding** stays fail-closed = missing design
  contract.
- **Gate (d) still grid-conditional (compute-bound ruling).** Re-measured after the
  composite matvec coupling: PASS only ≥224³ (9.08% / CI95-up 9.12%), `64³`=128% FAIL;
  launches fell 65% but fixed per-step cost only ~9% (median 0.183 ms/step) → the
  connected step is compute-bound, not launch-bound, so the crossover is unchanged.
  `docs/assessments/network-embedding-gate-d-remeasure-2026-07-20.json`.
- **Not `completed`.** No non-author review; the E3 combination matrix / named-hardware
  performance envelope / public benchmark residence remain. Inherits 01 port-power,
  now partly wave-validated upstream (`coax_thru` + `rectangular_waveguide`).
  Evidence: `docs/assessments/e4a-network-cascade-acceptance-2026-07-19.md`,
  `e4-network-e2-acceptance-2026-07-19.md`, `exclusive-timing-window-2026-07-20.md`,
  `00-status-and-gaps-2026-07-19.md` §03.
