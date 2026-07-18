# RF 端口、网络参数与天线工程闭环开发计划

状态：reopened（2026-07-18 从 archive 移回；契约层已落地，物理验证欠账）
日期：2026-07-14
初次标记完成：2026-07-15（形式验收；见下方复核结论）
复核结论：2026-07-18 审计判定实际证据等级为 E0–E1，**不满足声明的 E3**；详见 `docs/assessments/next-functional-audit-2026-07-18.md` 与 `docs/assessments/plan-01-rf-workflow-audit-2026-07-16.md`
优先级：下一阶段功能主线（当前欠账为全路线 P0）
目标证据：E3 production
执行范围：2026-07-15 按用户要求固定为单设备；跨设备并行、分片和归约不属于本计划的完成条件
依赖：当前 FDTD 可信度收敛工作持续满足本计划各 phase 的进入门；它不是本计划的功能 phase
Owner modules：`ports.py`、`scene.py`、`compiler/`、`fdtd/`、`postprocess/`、`result.py`
最近架构决策：2026-07-14，将 Lumped/Terminal/RF WavePort、RLC、N-port 和天线指标作为一个纵向项目

阶段验收：Phase 0-5 均已按独立 exit gate 验收，记录见
`docs/assessments/rf-workflow-phase-0-acceptance.md` 至
`docs/assessments/rf-workflow-phase-5-acceptance.md`。最终实现、测试和性能
门禁均限定为单 CUDA 设备。

## 1. 背景与当前能力

Maxwell 已有三维 GPU FDTD、`ModePort`、模式投影、Flux/Plane/ClosedSurface monitor、近远场、directivity 和一个基于通量的 S 参数后处理。公共架构已经稳定为 `Scene -> Simulation -> Result`，`SceneModule` 可把 PyTorch 参数物化成同一个 `Scene`。

当前能力还不能构成 RF 工程工作流：

- `ModePort` 是光子学模式端口，只能展开为 `ModeSource + ModeMonitor`；没有电压路径、电流环路、端子、参考面和端口终端语义。
- `compute_s_parameters()` 由 incident/transmitted flux 推导幅值，并明确把幅值放到复数实轴上；不能给出可信相位或完整 N-port 矩阵。
- 没有 Lumped/Terminal/Wave RF port，没有 R/L/C 负载和 accepted/available power。
- directivity 后处理只有在用户手工传入 `input_power` 时才计算 gain/radiation efficiency，无法自动连接馈电端口。
- 没有 Z/Y、重归一化、去嵌入、mixed-mode、VSWR、Touchstone 输出和标准端口诊断。

本计划把这些缺口作为一个产品能力交付，而不是拆成互相不可用的小 API。

## 2. 目标与非目标

### 2.1 目标

1. 交付 LumpedPort、TerminalPort 和 RF WavePort，明确电压、电流、方向、参考阻抗、参考面与功率约定。
2. 交付直接耦合 FDTD 的线性 R/L/C feed/load，所有逐步更新驻留 GPU。
3. 从一个或多个端口激励生成完整复数 N-port `NetworkData`，支持 S/Z/Y、重归一化、去嵌入和 mixed-mode。
4. 连接 accepted power 与远场，交付 gain、realized gain、辐射/失配/系统效率、轴比和共/交叉极化。
5. 保持 PyTorch tensor、autograd 和 `SceneModule` 工作流，不新增 backend-first 公共入口。
6. 为所有端口、网络、天线与梯度路径定义稳定的单设备执行和 device-preservation contract。

### 2.2 非目标

- 本计划不实现 SPICE/MNA、Touchstone 网络嵌入和非线性器件；它们分别由 `03`、`04`、`05` 计划承接。
- 不实现阵列 active S/MIMO、thin-wire、ferrite、SAR/热/静电/ESD。
- 不把现有 `ModePort` 改名或塞入 RF 特有分支；RF `WavePort` 是独立公共语义，但复用内部模式求解器。
- 第一版 RF WavePort 仅支持轴对齐平面和单导体/双导体常见截面；任意倾斜、弯曲和高度复杂多导体模式后置。
- 本计划所有端口、负载、网络、天线和梯度交付仅面向单设备 FDTD，不包含其他全波后端的实现或一致性验收。

## 3. 用户功能描述

用户应能在 `Scene` 中声明馈电与被动端口、放置 RLC 负载，运行单次端口响应或自动 N-port sweep，并直接从 `Result` 取得：

- 每个端口的复数 `V(f)`、`I(f)`、入射/反射波 `a/b`、输入阻抗与 accepted/incident/reflected/available power；
- 按 `[frequency, output_port, input_port]` 排列的复数 S，以及 Z/Y；
- return loss、insertion loss、VSWR、群时延、重归一化、参考面移动、mixed-mode；
- 天线 directivity、gain、realized gain、radiation/mismatch/system efficiency、axial ratio、co-pol/cross-pol；
- 可保存、可加载、可导出 Touchstone 的带单位、端口顺序和规范元数据对象。

失败必须尽早且可解释：端口未贴合 Yee 网格、路径闭合错误、端子短接、参考阻抗非正实、端口模式未收敛、激励频谱低于阈值、功率方向不一致或矩阵条件数过大时，`Simulation.prepare()` 或结果构造应报出端口名和具体原因。

## 4. Public API 草案

名称在 phase 0 API review 后冻结，语义先于名称。

```python
import torch
import witwin.maxwell as mw

feed = mw.LumpedPort(
    name="p1",
    positive=(0.0, 0.0, 0.8e-3),
    negative=(0.0, 0.0, 0.0),
    voltage_path=mw.AxisPath(axis="z"),
    current_surface=mw.Box(center=(0, 0, 0.4e-3), size=(1e-3, 1e-3, 0)),
    reference_impedance=50.0,
)

load = mw.TerminalPort(
    name="p2",
    positive_terminal=signal_pad,
    negative_terminal=ground_pad,
    integration_path=mw.AxisPath(axis="z"),
    reference_impedance=50.0,
    termination=mw.SeriesRLC(r=50.0, l=0.0, c=None),
)

scene = mw.Scene(domain, grid, boundary, ports=(feed, load), device="cuda")
scene.add_lumped_element(
    mw.Capacitor(name="matching_c", positive=node_a, negative=node_b,
                 capacitance=torch.tensor(0.8e-12, device="cuda"))
)

simulation = mw.Simulation.fdtd(
    scene,
    frequencies=torch.linspace(1e9, 10e9, 181, device="cuda"),
    excitations=mw.PortSweep(ports=("p1", "p2"), amplitude=1.0),
)
result = simulation.run()

p1 = result.port("p1")                 # PortData
network = result.network                # NetworkData
s11 = network.s[:, 0, 0]
z = network.to_z()
mixed = network.to_mixed_mode(pairs=(("p1", "p2"),))
network.renormalize(75.0).to_touchstone("device.s2p")

antenna = result.antenna(
    surface="nf2ff",
    driven_port="p1",
    polarization=mw.Ludwig3(),
)
loss = -antenna.realized_gain_db.max()
loss.backward()
```

公共约束：

- 所有 port 都通过 `Scene.add_port(...)`；RLC 可作为 port 的 `termination`，或通过 `Scene.add_lumped_element(...)` 放在两个端子之间。
- `Simulation.fdtd(..., excitations=...)` 仍返回 `Result`；`PortSweep` 是执行配置，不是新求解入口。单个主动端口可用 `PortExcitation("p1")`。
- `Result.port(name)` 返回单次/逐激励的 `PortData`；完整 sweep 才提供 `Result.network`，缺失激励列时不得用零值伪造完整矩阵。
- `NetworkData` 和 antenna 指标保持 `torch.Tensor`，只在显式文件 I/O 时转 CPU。
- `Scene.compile_ports(...)`、`Scene.compile_lumped_elements(...)`、内部 `compile_port_observers(...)` 遵守 compile-layer `compile_*` 命名；不恢复 `Scene.set_*`/`with_*` 或公开 `mw.FDTD`。

## 5. 物理约定与数据模型

### 5.1 端口约定

- 电压定义为从 negative 到 positive 的离散线积分 `V = integral E . dl`；方向反转同时改变 V 与 I 的符号，不改变功率。
- 电流由围绕 positive conductor 的右手定则闭合 H 环积分，或等价的端口面位移/传导电流通量取得。
- 统一使用峰值复相量，平均功率 `P = 0.5 Re(V I*)`；此约定写入结果元数据。
- power wave 默认采用 Kurokawa power-wave 定义，允许复参考阻抗；第一生产 gate 要求 `Re(Z0) > 0`。
- 固定功率恒等式：`P_incident = a^H a`、`P_reflected = b^H b`、`P_accepted = P_incident - P_reflected`；`gain = 4*pi*U/P_accepted`、`realized_gain = 4*pi*U/P_incident`、`EIRP = max(4*pi*U)`。任何不同 normalization 必须使用不同字段名。
- `available_power` 只有 `PortExcitation` 同时携带 generator/source impedance model 时才定义；仅有端口 `Z0` 和 V/I 时它为 `None`/unsupported，不能作为 `NetworkData` 的默认假设。
- 端口的 `reference_plane`、法向、terminal 顺序和去嵌入距离必须可序列化，不从几何位置隐式猜测。

### 5.2 公共结果对象

`PortData`：

- `frequencies: Tensor[F]`
- `voltage, current, a, b: complex Tensor[..., F]`
- `z_in: complex Tensor[..., F]`
- `incident_power, reflected_power, accepted_power: real Tensor[..., F]`；`available_power` 是带 generator provenance 的可选字段；
- `port_name`、方向、参考面、`z0`、单位与 phasor convention。

`NetworkData`：

- `frequencies: real Tensor[F]`
- `s: complex Tensor[F, N, N]`，矩阵索引固定为 `[out, in]`
- `z0: complex Tensor[F, N]`
- `port_names: tuple[str, ...]`
- `valid_columns: bool Tensor[N]`，禁止把未运行的列当作零；
- `metadata` 包含参考面、求解方法、激励谱阈值、归一化、端口 mode id 和收敛诊断；
- `to_z()/to_y()/renormalize()/shift_reference_planes()/to_mixed_mode()` 均返回新对象并保持 autograd graph；线性代数使用 `torch.linalg`。

`AntennaData`：角网格、`E_theta/E_phi`、radiation intensity、`P_rad/P_accepted/P_incident`、directivity/gain/realized gain、三种 efficiency、轴比和极化基；同时固定 `phase_center`、坐标 frame、polarization basis、power normalization provenance 和可选 excitation/basis 维。频率维始终显式，不对单频自动 squeeze。06 的 embedded element patterns 是标准 `AntennaData` excitation columns 的组合，不另建一套远场归一化。

共享 `PowerLossMonitor` / `PowerLossData` 是 RF、SIBC、wire、ferrite、SAR、Bioheat 和 ESD 的唯一耗散合同：按 channel 保存 conduction、electric/magnetic dispersion、nonlinear/circuit、surface、wire 和 total loss；volume/surface/line density 分别使用 W/m3、W/m2、W/m，积分量使用 W。数据必须携带 Yee 共点化方法、cell volume/face area/line length、occupancy/material/global ids、peak-phasor 的 `0.5` 时间平均约定、normalization/autograd provenance 和源 Result fingerprint。各功能只产生自己的 channel，不复制 monitor/result 类型。

### 5.3 内部编译对象

- `CompiledPortGeometry`：每个 Yee 分量的带符号稀疏权重、全局 index、owned slice 和方向。
- `CompiledPortExcitation`：源 waveform、drive kind、离散注入系数和主动列 id。
- `CompiledPortObserver`：V/I DFT accumulator、参考面和功率波变换。
- `CompiledLumpedElement`：拓扑端点、RLC 状态、更新系数和所影响的 E edges。
- `CompiledWavePortModes`：频率、传播常数、场模态、V/I 归一化、模式跟踪 id。
- `NetworkRunManifest`：端口顺序、每个 excitation 的 run id、归约状态和有效列。

## 6. 编译器、运行时、GPU-first 与 PyTorch 集成

### 6.1 编译路径

`Simulation.prepare()` 依次：解析端子/路径和网格 snapping；验证导体连通与边界间距；`compile_ports` 生成离散 V/I 权重；`compile_lumped_elements` 生成 RLC 状态；为每个激励生成 manifest；将所有 index、权重和状态一次性传到目标 CUDA device。几何拓扑检查可在 CPU 控制面执行，时间步内不得调用 Python/NumPy/CPU。

### 6.2 FDTD 耦合

- Lumped/Terminal source 与负载进入 Yee E 更新方程，不作为 `PointDipole` 的包装。
- R 通过局部导纳项隐式并入 E update；C 维护电荷/电压状态；L 维护支路电流状态。离散方案与 Yee leapfrog 对齐，并记录初始能量。
- V/I 与场在一致的 stagger 时间采样；DFT 相位补偿在 accumulator 内完成，不能事后以经验相位修正。
- 同一端口 source 与 observer 使用同一份离散几何权重，避免注入和测量路径漂移。
- 多端口 sweep 初版按输入端口独立 run；线性系统允许并行调度，但不使用同时激励后数值解混作为默认路径。

### 6.3 RF WavePort

截面模式求解器复用现有 mode core，但增加 conductor terminal 识别、modal V/I、characteristic impedance、功率正交归一化和跨频 mode tracking。WavePort 编译为等效场源与双向 modal observer；evanescent/未收敛模式不进入 S matrix，必须报告诊断。

### 6.4 PyTorch 和梯度

- R/L/C、参考阻抗、source amplitude 和支持的几何/材料参数允许为 device tensor；禁止编译时无条件 `float()`/detach。
- 端口 DFT、wave transform、S/Z/Y、de-embedding 和 antenna metrics 使用 torch 运算。
- FDTD adjoint 增加 port source/observer 与 RLC auxiliary state 的 replay/pullback；网络 sweep 的 backward 对有效 excitation 列求和。
- 离散端口拓扑、端口数量和 mode index 不可微；API 明确区分可微值与离散结构。

## 7. 单设备执行合同

本计划的运行、结果与梯度合同固定在一个显式选择的设备上。

- `NetworkRunManifest` 的 excitation 列按固定 port order 顺序执行并聚合，不依赖隐式设备迁移。
- 端口的稀疏 Yee index、V/I accumulator、RLC auxiliary state、WavePort mode fields、远场张量和 loss density 与所属场张量位于同一设备。
- 时间步内禁止 `.cpu()`、NumPy 转换、标量读取驱动控制流或其他 device-host round-trip。
- `PortData`、`NetworkData`、`AntennaData` 和 `PowerLossData` 只在显式文件 I/O 时创建 detached CPU payload；活跃结果保留原设备与 autograd graph。
- 任何输入 tensor 与 Scene 目标设备不一致时，在 `prepare()` 或结果构造阶段精确报错，不做隐式复制或 fallback。

## 8. Phases、依赖与 exit gates

### Phase 0：约定、API 与离散原型（E0, experimental）

交付：端口方向/phasor/power-wave 规范；`PortData/NetworkData` shape contract；LumpedPort V/I 离散原型；API review；benchmark golden-data 规范。
依赖：现有 field/flux DFT 相位约定可追踪。
Exit gate：均匀网格 TEM 单元中，离散 V/I 与解析值误差 `< 1%`；方向反转 invariance、峰值/RMS 因子和相位 stagger 测试全部通过；API 不引入第二求解入口。

### Phase 1：LumpedPort + RLC 一端口纵向切片（E1, experimental）

交付：LumpedPort、R/L/C、Series/ParallelRLC、PortExcitation、PortData、输入阻抗/return loss/VSWR/功率；FDTD forward 和持久化。
依赖：Phase 0。
Exit gate：50 ohm 匹配负载 `|S11| < -30 dB`（已做网格/窗口误差预算）；open/short 的幅值误差 `< 0.02`、相位误差 `< 3 deg`；串/并联 RLC 谐振频率误差 `< 2%`；全 run 无逐步 CPU transfer。

### Phase 2：TerminalPort + 完整 N-port NetworkData（E2）

交付：几何 terminal、自动逐端口 sweep、复数 S/Z/Y、重归一化、参考面移动、mixed-mode 和 Touchstone export。
依赖：Phase 1；`03` 只可在本 phase 的 `NetworkData` schema 冻结后开始集成。
Exit gate：同轴/微带二端口满足 reciprocity `max|S12-S21| < 0.02`、无源性最大奇异值 `<= 1.02`、功率不平衡 `< 2%`；Z/Y/S 往返相对误差 `< 1e-5`（well-conditioned 数据）；端口顺序和 run metadata 在重复执行间确定一致。

### Phase 3：RF WavePort（E2）

交付：axis-aligned coax/microstrip/stripline WavePort、多模 PortData、characteristic impedance、mode tracking 和去嵌入。
依赖：Phase 2；现有 mode solver 的高阶模式进入相应精度 gate。
Exit gate：同轴 `Z0` 解析误差 `< 2%`，矩形波导截止频率误差 `< 2%`，TEM 单模端口功率归一化误差 `< 1%`；跨频 mode id 无错误交换；LumpedPort 与 WavePort 在重叠有效频带的 S 参数差 `< 0.03`。

### Phase 4：天线工程结果（E2）

交付：accepted-power 自动连接、gain/realized gain/efficiencies、axial ratio、co/cross-pol、surface current，以及共享 `PowerLossMonitor/PowerLossData`。
依赖：Phase 2；WavePort-fed antenna 可在 Phase 3 后扩充。
Exit gate：自由空间半波偶极子峰值 directivity 与解析/高质量参考差 `< 0.25 dB`；`P_rad + P_loss` 相对 `P_accepted` 的误差 `< 3%`；realized gain 恒等式误差 `< 1e-5`；圆极化基准 boresight axial ratio 差 `< 0.5 dB`。

### Phase 5：单设备伴随闭环（E3）

交付：单设备端口/RLC adjoint、固定单模 WavePort adjoint、NetworkData/antenna 可微目标、文档与稳定 API。
依赖：Phase 1-4。
Exit gate：RLC、材料和支持几何参数的 adjoint 对至少三种步长的中心有限差分最佳稳定区相对误差 `< 2%`；固定单模 WavePort 的材料 adjoint 满足同一阈值；单设备未使用新功能的性能回退 `< 2%`。

## 9. 验收策略

### 9.1 单元与契约测试

- 路径/环路 orientation、grid snapping、端子短接/悬空/重名/越界验证。
- RLC 离散系数、能量和初值；所有 tensor dtype/device/shape 与序列化 round-trip。
- `a/b <-> V/I`、S/Z/Y、重归一化、mixed-mode、reference-plane shift 的已知矩阵测试。
- `valid_columns`、端口排序和保存加载不得静默变化。

### 9.2 数值与端到端

- open/short/matched load、串并联 RLC、无损线、衰减线、coax、microstrip、stripline、矩形波导。
- 检查 causal delay、reciprocity、passivity、power balance、网格/时间/运行长度收敛。
- 与解析传输线、独立 circuit 解、Tidy3D/XFdtd 或已审计 Touchstone reference 比较；reference 版本和生成参数入库。

### 9.3 梯度

- 对 R、L、C、材料介电率、端口附近几何参数验证 `S11`、accepted power、realized gain 梯度。
- 中心有限差分至少三种 step，报告 truncation/roundoff plateau；不只检查符号。
- 保存加载后的 detached inference 明确；活跃 autograd graph 不承诺跨文件持久化。

### 9.4 性能

- 记录 port observer 每 step 开销、每端口状态字节、每频率 DFT 字节、N-port wall time 和 GPU peak memory。
- 目标：一个 LumpedPort + 181 frequencies 相对同网格基础 FDTD 的 step-time 开销 `< 5%`；每增加一个被动 lumped port 的固定步进开销在代表网格上 `< 2%`。
- 无端口、无 RLC、无 RF monitor 的基础 FDTD 相对 Phase 0 前基线性能回退 `< 2%`，使用 CUDA event、预热和多重复中位数记录。

## 10. Benchmark 场景矩阵

1. `rf/lumped_open_short_match`：一端口校准、方向与功率。
2. `rf/series_parallel_rlc`：解析阻抗、谐振与 Q。
3. `rf/coax_thru`：Z0、传播常数、reference-plane shift。
4. `rf/microstrip_two_port`：S/Z/Y、损耗和 de-embedding。
5. `rf/differential_pair`：mixed-mode、mode conversion。
6. `rf/rectangular_waveguide`：TE10 截止、多模 WavePort。
7. `antenna/half_wave_dipole`：Zin、效率、gain。
8. `antenna/patch`：S11、surface current、realized gain、轴比变体。
9. `rf/single_device_no_feature_overhead`：无 RF 对象基础 FDTD 的 CUDA event 性能回归。

每个场景输出机器可读 metrics 并聚合到 `benchmark/RESULTS.md`；至少包含目标、当前值、reference provenance、网格和版本。

## 11. 主要风险与缓解

- **V/I 非唯一与路径依赖**：端口对象强制存储路径/面；只在准 TEM/定义良好的端子上提供阻抗，其他情况标记 modal/terminal definition。
- **时间 stagger 引入伪相位**：在 accumulator 内按采样时刻补偿，并用解析 delay gate 固定。
- **RLC 刚性导致不稳定**：采用被动、隐式离散并做离散正实性检查；不以减小全局 dt 掩盖错误。
- **高阶 mode 跟踪错误**：以 overlap + propagation constant 联合匹配，遇简并要求子空间匹配并输出置信度。
- **N-port 成本线性增长**：提供确定性逐列执行和 run cache；不牺牲可解释性做默认同时激励解混。
- **结果 API 膨胀**：保持三个核心对象 `PortData/NetworkData/AntennaData`，派生量为方法/属性，不用散乱 dict。

## 12. 完成定义

本计划完成需同时满足：Phase 0-5 的 exit gates；公共示例只使用 `Scene -> Simulation -> Result`；所有新增用户能力更新 `FEATURE_LIST.md`；四类 canonical workflow（匹配传输线、RLC 网络、RF WavePort、馈电天线）可从全新进程复现；GPU profiler 无时间步 CPU round-trip；支持项有梯度与单设备明确合同，不支持项精确报错；benchmark 当前结果达到本文阈值并进入 CI 的分层回归集。
