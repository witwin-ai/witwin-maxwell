# Thin-Wire 亚网格导线模型开发计划

> 状态：active；Phase 0-3 accepted，Phase 4 pending
> 日期：2026-07-14
> 目标证据：E3 production
> 类型：独立数值能力，不是几何体素化便利功能
> 前置依赖：现有 Yee FDTD、非均匀网格、PEC/导电材料；RF 端口仅在后期端接集成时需要
> Owner modules：拟新增 wire 场景/编译模块、`compiler/`、`fdtd/`、`monitors.py`、`result.py`
> 最近架构决策：2026-07-14，采用能量一致的辅助 wire current/charge 网络与离散 incidence 耦合，不以细圆柱体素化作为实现
> 公共架构约束：`Scene + Simulation + Result`，GPU-first

## 1. 背景与当前能力

Maxwell 可以把圆柱或其他导体几何体体素化为 PEC/有限电导率材料，但当导线半径 `a` 远小于局部网格间距 `Δ` 时，这种表示会把半径量化成网格尺寸，无法保持正确的每单位长度电感、电容、表面损耗、输入阻抗和共振频率。简单地强制网格解析皮肤深度或半径会使天线、线缆和 PCB bonding wire 的三维场景不可承受。

当前没有一等 thin-wire 数据模型、线图拓扑、端点/连接语义、亚网格电流状态或专用 CUDA 更新。XFdtd gap analysis 明确指出“体素化细导线不能替代亚网格细线模型”。

## 2. 目标与非目标

### 2.1 目标

1. 在 `a << Δ` 时仍以物理半径控制导线的电磁响应，而不是以体素宽度代替半径。
2. 支持 PEC 直线、折线、分支、闭环、开放端点和接地端点；后续支持有限电导率/表面阻抗。
3. 导线与 Yee 场之间使用离散互易的采样/沉积算子，保证无源条件下的离散能量稳定性。
4. 支持 uniform/custom/auto grid，并定义导线跨 cell、交点、端点和端口的唯一 ownership。
5. 导线电流、电荷、损耗和端点量可监测、序列化并进入端口/天线结果链。
6. 提供可用于半径、路径和材料优化的 PyTorch 合同，以及单机 multi-GPU 分片合同。

### 2.2 非目标

- 不以自动极细网格替代亚网格模型。
- 不在第一版处理导线绝缘层、绞线、编织屏蔽、趋肤/邻近效应的完整横截面场。
- 不把 thin wire 当成任意 1D transmission-line/cable circuit；多导体传输线宏模型另立计划。
- 不承诺半径接近或大于局部 cell 时仍使用 thin-wire 模型；该区域应使用解析几何体素化。
- 不在 Phase 1 支持任意斜向段、有限电导率和复杂 junction；先以轴对齐 PEC 建立可信纵向切片。
- 不增加 CPU solver fallback。

## 3. 用户功能描述

用户以物理中心线、半径和导体模型定义导线。`Scene` 在 prepare 时将其编译成稳定的线图和 Yee 耦合，而不是把它转换为小圆柱：

```python
import witwin.maxwell as mw

wire = mw.ThinWire(
    name="dipole",
    points=((0, 0, -0.24), (0, 0, 0.0), (0, 0, 0.24)),
    radius=5e-4,
    conductor=mw.WireConductor.pec(),
    endpoints=(mw.WireEnd.open(), mw.WireEnd.open()),
)

scene = mw.Scene(...)
scene.add_thin_wire(wire)
scene.add_monitor(mw.WireMonitor(name="wire_state", wire="dipole", frequencies=freqs))
result = mw.Simulation.fdtd(scene, frequencies=freqs).run()

result.monitor("wire_state").current
result.monitor("wire_state").charge
result.monitor("wire_state").ohmic_loss
```

当 RF port 功能可用后，用户可把一个 wire node 或 gap 绑定到标准 `LumpedPort`/`TerminalPort`，但端口仍由 `Scene.add_port(...)` 管理，thin-wire 不创建另一套 source API。

## 4. Public API 草案

### 4.1 场景对象

- `ThinWire(name, points, radius, conductor, endpoints=None, snap=...)`：不可变物理定义；`points` 为绝对 SI 坐标，`radius` 可为标量或逐段张量。
- `WireConductor.pec()`；后期增加 `WireConductor.finite(conductivity, permeability=...)` 和 `WireConductor.surface_impedance(...)`。
- `WireEnd.open()`、`WireEnd.grounded(structure=...)`、`WireEnd.node(name=...)`；junction 由共享命名 node 或显式 `WireJunction` 形成。
- `WireMonitor(name, wire, frequencies=..., quantities=(...))`。
- `Scene.add_thin_wire(...)` 和只读 `Scene.thin_wires`；不通过 `add_structure` 假装它是有体积 `Structure`。

### 4.2 编译和结果

- `Scene.compile_thin_wires()`：返回 torch-native `CompiledWireNetwork`；
- `Scene.compile_wire_monitors()`：解析采样；
- 内部 `compiler/thin_wire.py::compile_thin_wires(...)`；
- `Result` 通过普通 monitor 访问 `WireData`；端口绑定时，compiled wire provider 向 01 的标准 `Result.port(name) -> PortData` 路径提供同一 voltage/current 离散量，不建立 `Result.ports` 容器或第二套端口结果。

API 必须验证：点数、零长度段、重复 node、半径正值、相交拓扑、导线与 PEC 接触、PML 禁入区域、局部 `a/Δ` 有效区间和过近的平行导线。

## 5. 数据模型与离散合同

### 5.1 物理线图

`CompiledWireNetwork` 至少保存：

- node 坐标、segment 端点索引和有向 incidence matrix `B`；
- 每段物理长度、半径、电导率/表面参数和局部切向；
- segment 到 Yee E 边的采样矩阵 `G`，以及严格转置的 wire-current 到 Ampere 方程沉积 `G^T`；
- node 电荷、segment 电流、历史/ADE 状态；
- 端点条件、junction、端口绑定和 monitor 索引；
- grid fingerprint、有效性 warning 和 ownership。

### 5.2 数值表示选择

采用**能量一致的辅助线电流/电荷（auxiliary wire current/charge）耦合**，而不是只把若干 E 边设为零：

1. 线电流 `I` 作为 segment 状态，通过保守沉积进入离散 Ampere 更新；
2. 线电荷 `q` 通过 `dq/dt + B I = 0` 满足离散连续性；
3. Yee E 沿中心线的离散线积分通过 `G E` 驱动 wire update；沉积使用其转置，确保场对 wire 所做功与 wire 从场获得的功相消；
4. 物理半径通过解析的对数近场/有效半径修正进入局部 self term，而非由 `Δ` 替代；
5. 时间离散采用与 leapfrog 对齐的梯形/隐式局部更新，使 PEC/无损导线不会产生数值能量。

Phase 0 必须写出离散能量式和稳定性界，并用一个小型 torch reference 实现比较候选的 Holland/contour-path effective-radius 与辅助线状态离散。只有满足能量、半径不变性和非均匀网格合同的方案才能进入 CUDA；本文不允许在推导前把经验系数固化成公共 API。

### 5.3 有效范围

编译器计算每段 `a/Δ_perp` 和相邻导体距离。目标有效带为亚网格区，例如 `a/Δ_perp <= 0.2`；最终阈值由 benchmark 冻结。超出上限建议使用体几何；半径过小导致 log self term 病态、或两根导线落入相同不可分辨 stencil 时必须报错或显式 warning，不能静默给出网格半径结果。

## 6. 编译器与原生 CUDA 运行时方案

### 6.1 编译流程

1. 规范化 polyline、合并容差内 node、检测相交和分支。
2. 每个 physical polyline segment 编译为 cell-local `WireFragment`，按穿越的 Yee edge/cell 切分并保持总物理长度；Phase 3 对斜向 fragment 建立 conservative edge weights。
3. 计算每段局部横向 primal/dual spacing 和 effective-radius self coefficients。
4. 生成按 segment/node 压缩的一维 SoA 张量，不创建全 3D wire state。
5. 建立 `G` 的紧凑 index/weight 列表，并验证采样/沉积转置关系。
6. 对与 PML、周期/Bloch wrap、PEC、端口重叠的情况进行物理检查。

### 6.2 每步推进

推荐时序：H 更新 → wire 驱动场采样 → `I/q` 局部更新 → wire current 沉积到 E update → E/边界更新。具体半步位置必须以离散能量推导为准，并在 forward/reverse 使用相同 schedule。

CUDA 增加三个窄职责 kernel：

- `sample_wire_emf`：按 segment 对 E 线积分；
- `update_wire_state`：推进 `I/q` 和有限电导率/ADE 状态；
- `deposit_wire_current`：按转置权重写入 E 更新 RHS。

对于共享 Yee edge 的多个 segment 使用排序后的 segmented reduction，避免不确定性 atomics；规模小的无冲突网络可走直接 kernel。CUDA graph capture、checkpoint 和 restart 必须包含 wire state。

### 6.3 有限电导率

Phase 4 以每单位长度串联阻抗 `Z'(omega)` 表示损耗。低频均匀电流和高频 skin-effect 使用各自有效公式，跨频带模型拟合为被动 rational ADE；若已经完成通用表面阻抗计划，则复用其正实拟合器和状态空间，不复制 pole fitting。

## 7. PyTorch/autograd 合同

- node/segment coefficients、monitor current/charge 和所有后处理保持 torch-native。
- Phase 1 只承诺场对 wire 激励以及 monitor 输出的 forward；Phase 2 实现 wire recurrence 的精确反向转置。
- 半径、连续 node 坐标、导电率和端口波形可成为 tensor 参数。拓扑、segment 数、snap path 和 pole order 是离散编译决策，不可微。
- 几何坐标梯度仅在固定 segment-to-grid stencil 内有效；跨 cell 时梯度不连续，必须由 `SceneModule` 文档和 runtime metadata 标明。
- gradient test 使用 float64 torch reference/高精度有限差分；CUDA float32 给出相对/绝对混合容差。
- reverse checkpoint 必须保存或可重放 `I/q` 以及全部 surface/ADE state，禁止只反演 Yee fields。

## 8. Multi-GPU contract

- `WireFragment` 按其耦合 Yee component owner 分配；physical segment/node state 由稳定 global id 选择唯一 owner，node 规则与端口 node 共用。
- 长 physical segment 可以跨任意多个 shard；每个 fragment 只采样本地 owned/普通 stencil ghost，不能假设单层 halo 覆盖整段。fragment EMF/current contribution 通过端口式标量/稀疏 reduction 汇总给 state owner。
- `I/q` 只有 state owner 推进；沉积到邻 rank owned E edge 的 contribution 进入 send buffer，在 E update 前按 component owner 加和，不允许多个 rank 各自推进同一 physical segment。
- wire junction continuity 需要边界 node 电流和的邻居交换，数据量与跨界 segment 数成正比。
- monitor 电流/损耗按 segment owner 局部累积，再做 deterministic reduction。
- reverse 使用完全相反的通信和转置沉积；单/多 GPU 必须通过 value、能量和 gradient parity。

## 9. Phases、依赖与 Exit Gates

Phase 0 必须冻结 `AcceptanceBudget`。除非某场景在表格中显式覆盖，默认 gate 为：torch/reference `rtol<=1e-5`；解析/NEC 核心阻抗或传播量相对误差 `<=2%`；能量/电荷残差 `<=1%`；至少三档网格/时间步报告收敛；支持参数梯度相对误差 `<2%`；multi-GPU 继承 02 的 field/monitor/power parity；无 wire 场景性能回退 `<1%`。定性“通过/一致”均指满足该预注册预算。

### Phase 0：数值推导与 reference prototype（E0, accepted 2026-07-15）

交付物：离散方程、能量式、有效半径修正、稳定性条件；轴对齐单线 torch reference；候选方法决策记录。

Exit gate：在 `a/Δ` 跨至少一个数量级时，解析每单位长度电感/电容和波传播不随体素宽度漂移；无损闭合线的离散能量不增长。

### Phase 1：轴对齐 PEC 单线 GPU forward（E1, accepted 2026-07-15）

交付物：`ThinWire`、`add_thin_wire`、编译器、压缩状态、CUDA forward、开放/接地端点、`WireMonitor`。

依赖：无 RF port 依赖。

Exit gate：直线、L 形和半波偶极子 forward 通过；半径小于 cell 时结果跟随物理半径；CUDA 与 reference parity；不得出现 CPU fallback。

### Phase 2：网络拓扑与 adjoint（E2, accepted 2026-07-15）

验收证据：`docs/assessments/thin-wire-phase-2-acceptance.md`。

交付物：折线、junction、branch、closed loop；端点电荷守恒；wire state checkpoint；半径/材料参数 adjoint。

Exit gate：每个 node 的离散连续方程残差满足容差；闭环/分支能量守恒；梯度通过有限差分。

### Phase 3：任意方向、非均匀网格与端口绑定（E2, accepted 2026-07-16）

验收证据：`docs/assessments/thin-wire-phase-3-acceptance.md`。

交付物：斜向 conservative coupling；custom/auto grid；周期边界合法路径；`LumpedPort`/`TerminalPort` gap/node 绑定。

依赖：RF 基础端口合同已冻结。

Exit gate：旋转同一 dipole 的输入阻抗/远场只呈现网格收敛误差；端口 accepted power 与 wire/辐射/损耗平衡闭合。

### Phase 4：有限电导率、宽带损耗与 multi-GPU（E3）

交付物：finite conductor/rational skin effect；ohmic loss；跨 partition wire；单/多 GPU reverse parity；性能优化。

依赖：共享 rational-model 基础设施；空间 multi-GPU gradient 还依赖 02 Phase 7 加入 wire reverse communication。此前只承诺单 GPU adjoint，并在 trainable joint-solve prepare 时拒绝。

Exit gate：导线 AC resistance 与解析 skin-effect 曲线一致；多 GPU value/gradient/energy parity；压缩 state 显存与 segment 数线性增长。

## 10. 验收策略

### 10.1 单元测试

- polyline、node merge、交点、零长度、半径和 PML 检查；
- segment 切分长度守恒；`G/G^T` 内积测试；
- continuity、checkpoint round-trip、序列化和 deterministic reduction；
- 编译 cache 对路径/半径/网格变化正确失效。

### 10.2 解析测试

- 细直导线每单位长度电感/外部磁能的对数半径规律；
- 两线/同轴准 TEM 的 `Z0` 和传播速度（在模型有效几何范围内）；
- 短偶极子辐射电阻和 `sin(theta)` 远场；
- 半波 PEC dipole 的共振频率、输入电阻和方向性；
- 有限导体的 DC/skin-effect AC resistance。

### 10.3 独立求解器

- NEC-2/NEC-4 或其他 MoM：直线 dipole、folded dipole、monopole、loop 的输入阻抗和远场；
- 至少一个公开测量天线案例；
- 对半径已被体网格解析的中间区，与普通圆柱 PEC FDTD 交叉比较，确认两个模型在重叠有效区收敛。

### 10.4 收敛、能量与性能

- 固定物理半径做 `Δ` 收敛；固定 `Δ` 做 radius sweep；旋转角和端点位置 sweep；
- 无损场景检查离散总能量和 continuity；有损场景检查输入 = 辐射 + wire ohmic + 其他材料损耗；
- 记录每 10 万 segment-step 的时间、peak memory、atomics/reduction 占比；
- 没有 thin wire 的场景必须保持零运行时开销和零额外 state 分配。

## 11. Benchmark 矩阵

| 场景 | 主要指标 | 参考 |
| --- | --- | --- |
| 无限/长直线局部截面 | 对数 self term、L' | 解析 |
| 短 dipole | 辐射电阻、远场 | 解析 |
| 半波 dipole 半径扫描 | 共振、输入阻抗、gain | NEC/MoM |
| monopole over PEC | 输入阻抗、方向图 | image theory + NEC |
| loop / folded dipole | junction continuity、阻抗 | NEC/MoM |
| 有限电导率 wire | AC resistance、效率 | 解析 skin effect |
| 跨 GPU 长线/网络 | parity、scaling | 单 GPU baseline |

场景进入 `benchmark/scenes/thin_wire/`，保留原始复阻抗、远场和收敛表，不只保留 PNG。

## 12. 风险与缓解

- **错误的亚网格公式只拟合单一案例**：Phase 0 要求能量推导、半径 sweep 和多个独立案例后才进入公共 API。
- **斜线网格各向异性**：轴对齐先交付，斜向方案必须通过旋转不变性 gate。
- **junction 产生非物理电荷**：incidence matrix 和 continuity 残差作为每步可调试诊断。
- **thin-wire 与端口/SIBC 重复计损耗**：编译器明确 ownership；同一段只能选择一种 conductor law。
- **atomics 导致不确定梯度**：预排序 segmented reduction 和稳定 owner 规则。
- **超出模型适用范围**：结果写入 `validity` 元数据；明显无效配置拒绝而非静默体素化。

## 13. 完成定义

1. `ThinWire` 是一等 Scene 对象，物理半径在亚网格区真实影响结果。
2. PEC/有限导体、直线/折线/junction、端点和 RF 端口绑定均走同一 compiled wire network。
3. CUDA forward、adjoint、checkpoint 和 multi-GPU 没有 CPU side path。
4. 解析、NEC/MoM、体几何重叠区、网格/半径/旋转收敛、能量/电荷守恒全部通过。
5. 无 wire 场景无性能回归；有 wire 的内存随 segment 数而非 3D cell 数增长。
6. 公共示例遵守 `Scene -> Simulation -> Result`，并同步更新 `FEATURE_LIST.md`。
