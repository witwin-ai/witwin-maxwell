# MoM 开放表面、PEC 与薄导体开发计划

> 状态：draft；Phase 0a（外部 `PointDipole` 源）已实现；Phase 0、1、3 已在分支 `sie/open-pec-efie` 实现（待合并）；其余阶段未开始
> 日期：2026-10-28
> 类型：MoM 数值能力扩展（新 formulation 与新拓扑），不是几何便利功能
> 前置依赖：现有 RWG / PMCHWT / Duffy 奇异积分、`DenseOperator` 与 Riesz 度量、H/H2/FMM 后端
> Owner modules：`mom/mesh.py`、`mom/operator.py`、`mom/quadrature.py`、`mom/physics.py`、`mom/linalg.py`、`mom/fields.py`，以及 fast operator 模块
> 公共架构约束：`Scene + Simulation + Result`，GPU-first，不新增 CPU fallback
> Bench 约束：任何 `witwin/maxwell/mom/*.py` 的改动都需要新的 provider epoch（见 Maxwell-Bench 的 epoch gating 流程），不允许原地修改已冻结的 gate

## 1. 背景与当前能力

当前 MoM 只支持**互不相交、不嵌套、闭合、均匀、无源（passive）的介质体，背景为真空**：

- `SurfaceMesh` 要求闭合二维流形：每条边恰好两个面，拒绝非流形顶点（`mom/mesh.py` 的 `_topology`）。
- `RWGBasis` 在所有边上建立基函数，没有边界边或 junction 边的概念。
- 只有 PMCHWT 一种 formulation：未知量为 J/M，矩阵为 `2ne × 2ne`（`mom/operator.py` 的 `_assemble_scalar_regions`，`formulation_id = pmchwt_exp_minus_iwt_total_trace_v1`）。
- `physics.py` 拒绝 `Material.pec`，也拒绝 `thin_wires`、`ports`、`material_regions` 等场景特性。
- 生产计划 `mom_generalized_production_development_plan.md` 明确把开放表面、嵌套/多层体和几何梯度列在范围之外。

因此以下场景目前都无法求解：零厚度铜片、PCB 走线和贴片、FSS/RIS 单元、闭合 PEC 体、电阻膜或石墨烯等 2D 薄层，以及铜贴在介质衬底上的结构。

FDTD 侧已经有对应的材料模型（`Medium2D`、`Graphene`、`LossyMetalMedium`、`Material.pec()`），MoM 应复用同一组公共 `Material` / `Structure`，不另建一套 API。

## 2. 物理背景与设计依据

### 2.1 薄铜的工作区间

铜（σ = 5.8e7 S/m）的趋肤深度 δ = 1/√(π f μ0 σ)：

| 频率 | δ |
| --- | --- |
| 1 GHz | 2.1 µm |
| 10 GHz | 0.66 µm |
| 28 GHz | 0.39 µm |
| 100 GHz | 0.21 µm |

PCB 铜厚 17–35 µm，满足 δ ≪ t ≪ λ。正确的模型是零厚度表面加上表面条件，而不是把它当成薄闭合体：

- 闭合薄体（上下两面加侧边）的上下表面相距只有 t，会产生近奇异积分和 thin-body breakdown；侧边的网格尺寸需要接近 t，未知量数会暴涨。
- 零厚度开放表面上的未知量是**总电流** J = J_top + J_bot。这是 MoM 处理贴片、FSS、走线的标准做法。

### 2.2 表面条件的选择

| 情况 | 条件（`exp(-iωt)` 约定） | 来源材料 |
| --- | --- | --- |
| 无损良导体，只关心场/阻抗 | PEC：n × E = 0 | `Material.pec()` |
| 零厚度导电薄层（t ≪ δ）、电阻膜、2D 材料 | 阻抗薄层：E_tan = Z_sheet · J，Z_sheet = 1/σ_s | `Medium2D(sigma_s=...)`、`Graphene` |
| 薄导体，厚度在 δ 附近 | Z_sheet(t) 由导电板的传输矩阵导出；两端极限：t ≪ δ 时 → 1/(σt)，t ≫ δ 时 → Z_s/2（上下两面并联） | 新增参数化，或 `LossyMetalMedium` 加厚度 |
| 闭合良导体，需要铜损 | Leontovich：E_tan = Z_s (n × H)，Z_s = (1 − i)√(ωμ0/(2σ)) | `LossyMetalMedium` |

单层阻抗模型只描述上下两面的对称（总电流）模式。对于 t ≫ δ 的铜，这足以给出损耗和场，但无法分辨上下两面各自的电流分配。需要这个分配的场景（例如贴近地平面的走线下表面电流集中）留到 Phase 6 用多区域模型处理。

### 2.3 棱边奇异性（Meixner 边缘条件）

奇异性来自物理本身，与离散方法无关；MoM 的优势在于棱边几何是精确的，没有阶梯近似。

- 薄 PEC 板边缘：平行于边的电流和电荷密度 ∝ r^(-1/2)，垂直于边的电流 ∝ r^(1/2) → 0，场 ∝ r^(-1/2)。
- PEC 外角为 270°（直角块的棱）：场 ∝ r^(-1/3)。
- 介质棱角：奇异性较弱，指数随对比度变化。

后果：使用均匀 RWG 网格时，积分量（远场、RCS、输入阻抗、S 参数）收敛良好，但**棱边附近的近场收敛很慢**。验收时必须把积分量和近场分开报告。

### 2.4 条件数

- 开放表面上 MFIE 和 CFIE 都不成立，只能用 EFIE。EFIE 是第一类积分方程，有两类 breakdown：
  - 低频 breakdown：矢量位与标量位两项的尺度相差 (kh)²；
  - 网格加密 breakdown：条件数 ∝ 1/h²。
- 闭合 PEC 用纯 EFIE 或纯 MFIE 都会在内腔谐振频率失效，必须用 CFIE。
- 补救手段：Calderón 预条件（用 Buffa–Christiansen 对偶基构造 EFIE 的自预条件），或准 Helmholtz / loop-star 投影。
- 残差度量：EFIE 残差小**并不**意味着误差小。所有残差（包括 fallback 判据、给神经网络用的物理损失）必须在 Calderón 预条件之后、或在 Riesz（对偶）范数下计算。现有的 `CompactRWGRiesz` 和 `apply_test_riesz_inverse` 是起点。

## 3. 目标与非目标

### 3.1 目标

1. 零厚度开放 PEC 表面（板、带、贴片、FSS 单元），采用 EFIE。
2. 闭合 PEC 体，采用 CFIE；可与现有 PMCHWT 介质体在同一场景共存（互不相交）。
3. 阻抗薄层（`Medium2D`、`Graphene`、厚度参数化薄导体），以及闭合良导体上的 Leontovich 条件。
4. 棱边附近近场的可控精度：网格向边缘渐变加密，可选带 Meixner 奇异性的边缘基函数。
5. 在频率和网格尺寸变化时保持迭代收敛，即实现 Calderón 预条件或准 Helmholtz 投影。
6. Junction：多个表面共享一条边（T 形连接），以及 PEC 片贴在介质体表面（铜在衬底上）。
7. 新 formulation 接入 H/H2/FMM：EFIE 只需要单层算子 L。
8. 导出神经求解器需要的量：每条 RWG 边的电流系数、预条件后的残差算子、边缘奇异权重（见第 8 节）。

### 3.2 非目标

- 不支持体积分方程意义下的非均匀衬底；分层介质 Green 函数另立计划（Phase 8）。
- 第一版不做几何梯度。
- 不做过孔或细线与表面的 junction；与 `07-thin-wire-model.md` 协调后另立计划。
- 不做端口、S 参数和电路耦合；在 Phase 5 之后单独立项。
- 不做时域 MoM。
- 不新增 CPU solver fallback。

## 4. 用户可见功能

- `Structure(geometry=<零厚度 Box 或开放三角网格>, material=Material.pec() | Medium2D(...) | Graphene(...))` 可以直接交给 MoM 求解，不需要另外的 API。
- `Structure(geometry=<闭合几何>, material=Material.pec() | LossyMetalMedium(...))` 走 CFIE 或 Leontovich 路径。
- `Simulation` 的 MoM 配置增加 formulation 和预条件器选项；不支持的组合必须显式拒绝（fail closed），不能静默退化。
- `Result` 提供表面电流（J，介质体为 J/M）、任意点的场，以及远场和 RCS。
- 待确认：`witwin.core` 的 Mesh 几何是否允许开放网格。如果不允许，在 core 新增开放表面几何类型，FDTD 体素化和 MoM 共用这一类型。

## 5. 数值设计

### 5.1 拓扑

- 新增表面拓扑分类：内部边（2 个面）、边界边（1 个面）、junction 边（≥ 3 个面）。
- 开放表面各自要求方向一致（可定向），但不要求闭合；闭合介质体的验证保持不变。
- 每个 body 记录类型（介质闭合 / PEC 闭合 / PEC 开放 / 阻抗薄层），以及它和其他 body 的 junction 关系。
- 拓扑检查放到 GPU 上做（现有 `_topology` 在 host 端），这一点和 Phase 0 一起完成。

### 5.2 基函数

- 开放表面：只在内部边上建立 RWG。边界边不配基函数，等价于强制电流法向分量为零。
- Junction 边：N 个面共享一条边时，建立 N − 1 个独立的 junction 基函数（以一个面为参考，与其余各面配对）。
- PEC 与介质的 junction：介质表面上被 PEC 覆盖的部分 M = 0，只保留 J；PEC 片与介质表面相接的边需要专门的 junction 基函数（多区域 junction formulation）。
- 可选：边界边相邻的三角形使用 Meixner 加权基函数，形式为 J ∝ r^(-1/2) × 光滑部分。

### 5.3 积分

- 自作用项、共边和共点的 Duffy 规则可直接复用（`duffy_fan`、`touching_pair_rule`）。
- 边界边附近的观察点使用已有的 `distance_scaled_duffy_fan(..., edge_grading=True)`，并为开放边补充收敛测试。
- 薄间隙：上下两层很近的平行开放表面（例如多层 PCB）需要按 间距/尺寸 比例自适应的近奇异规则；现有 `near_ratio` 分类要加入间距判据。
- 场点距离表面小于网格尺寸时的近场求值，要在 `fields.py` 中单独实现奇异减除积分。

### 5.4 Formulation

| Identity | 方程 | 未知量 |
| --- | --- | --- |
| `efie_open_pec` | (L J)_tan = −E_inc,tan | J |
| `cfie_closed_pec` | α·EFIE + (1 − α)·η·MFIE，α 默认 0.2–0.5 | J |
| `efie_impedance_sheet` | (L J)_tan − Z_sheet · J = −E_inc,tan（Z_sheet 进入 Gram 矩阵项） | J |
| `pmchwt_leontovich` | 闭合良导体，M = −Z_s (n × J) 消去 M | J |
| `multiregion_junction` | PMCHWT 介质区域 + PEC/阻抗片 junction 耦合 | J/M（介质），J（导体） |

每个 identity 有独立的 `formulation_id` 和版本号；与 kernel、后端标识正交，沿用现有的 receipt 结构。

### 5.5 预条件与残差

- 优先实现 Calderón 预条件（BC 对偶基，用于 `efie_open_pec` 和 `efie_impedance_sheet`）。开放表面的 Calderón 需要处理边缘（使用修正的对偶基）；如果周期过长，先用准 Helmholtz 投影作为过渡。
- `DenseOperator` 和 Krylov 协议增加"预条件残差"接口；`residual_loss`、`condition_diagnostics` 报告预条件前后两种残差和条件数。
- CFIE 的 α 和 MFIE 的恒等项测试（rotated-RWG 测试空间）沿用 PMCHWT 的约定，并记录在 receipt 中。

### 5.6 快速算子

- EFIE 只用单层算子 L（矢量位项加标量位项）。H/H2/FMM 要支持只装配或只作用 L 块，而不是整个 PMCHWT 的 2×2 块。
- Capacity 估计和版本 guard 要为每个新 formulation 单独登记。

## 6. 分阶段计划

每个阶段都要有显式的拒绝路径；不支持的组合不能静默运行。

| Phase | 内容 | 交付与验收 |
| --- | --- | --- |
| 0a | 现有 PMCHWT 增加外部 `PointDipole` 源（目前只支持 `PlaneWave`）；偶极子必须位于所有 body 外部 | 解析入射场与 RHS；介质球用 Mie 偶极子解检查；与 FDFD（高 ppw）交叉检查。这是神经求解器 MoM 路线的最早依赖，可以独立于 Phase 0 先做 |
| 0b | **棱角体与薄体作为一等目标**。现状：公共路径只接受 `Sphere`、`Ellipsoid` 和显式闭合 `Mesh`，`Box`/`Cylinder` 被拒绝；内置网格只有球、椭球、球谐变形体；验证几乎只做过光滑体。内容：`Box`（含旋转）、`Cylinder`、`Cone`、圆角立方体族、棱柱、薄闭合板的表面网格化，提供向棱边和角点渐变加密的选项；公共 `Scene -> Simulation -> Result` 路径直接接受这些 primitive | 网格拓扑和质量测试；在立方体、薄板上测量均匀网格与渐变网格的收敛阶；积分量与棱边附近（λ/20 以内）的近场分开报告；参考：轴对齐立方体用高 ppw FDFD 加 Richardson 外推，同一网格用外部开源 MoM 交叉检查；圆角半径扫到 0 检验连续性 |
| 0 | 拓扑分类、开放表面 RWG、body 类型、`physics.py` scope 检查改为按 formulation 判定 | 拓扑单元测试（板、T 形 junction、带孔板、非流形拒绝）；现有闭合 PMCHWT 结果逐位不变 |
| 1 | `efie_open_pec`，dense 后端，真空，场与 RCS | 方板 RCS 的网格收敛；圆盘与高阶旋转对称参考解比较；带状偶极子输入阻抗；互易性；无损功率守恒（光学定理） |
| 2 | `cfie_closed_pec`；PEC 闭合体与介质 PMCHWT 体同场景共存 | PEC 球 Mie 解（含内腔谐振频率点，用来证明 CFIE 无谐振）；PEC 与介质混合多体场景的互易性 |
| 3 | Calderón / 准 Helmholtz 预条件，预条件残差接口 | 迭代次数对 h 和频率基本不变（频率从 0.01λ 到几 λ 尺寸扫描）；预条件残差与真实误差的相关性测试 |
| 4 | 边缘渐变网格与可选 Meixner 基；近场奇异减除求值 | 边缘电流指数拟合（−1/2 与 1/2）；边缘附近 λ/20 内近场的收敛阶报告，与积分量分开 |
| 5 | `efie_impedance_sheet`（`Medium2D`、`Graphene`、厚度参数化）与 `pmchwt_leontovich` | 无限阻抗薄层的解析反射：正入射 Γ = −η0/(η0 + 2Z_sheet)；t → 0 时闭合薄体收敛到开放片；铜损功率与解析值比较 |
| 6 | Junction：T 形 PEC junction；PEC 片贴在有限介质体表面（有限衬底贴片） | 电流连续性（junction 处 KCL 残差）；有限衬底贴片的谐振频率和输入阻抗与本仓库 FDTD PCB fixture 交叉验证 |
| 7 | 新 formulation 接入 H/H2/FMM | 与 dense 的逐元素误差；规模扫描中的内存和时间；版本 guard |
| 8 | （另立计划）分层介质 Green 函数 kernel、过孔和细线 junction、端口与 S 参数 | — |

Phase 0a 状态（2026-10-28，分支 `sie/point-dipole-source`）：已实现。`PointDipole(profile="ideal")` 作为理想电流矩 `Il` [A·m]，解析入射 E/H（含全部近场项）进入现有解析 incident-trace lifting，formulation id 不变；dipole carrier 使用独立的版本化 identity，PlaneWave carrier identity 逐位不变。只支持 linear dense free-space 标量运行时；偶极子位于 body 内部/表面、任一面元距离小于 3 倍面元半径、积分阶数低于 4、Gaussian profile、非 CW、磁/分布电流以及 H/H2/FMM、周期、张量、谐波平衡与体耦合配置都显式拒绝。与独立的矢量球面波偶极子-球级数参考比较（含有耗介质、1.6–2.5 倍半径距离），网格每细分一次误差约降为 1/4，ka=1、1280 面元时外场/内场误差 0.93%/0.48%。与 FDFD（高 ppw）交叉检查尚未完成；Bench 使用前需要新的 provider epoch。

Phase 0/1/3 状态（2026-10-29，分支 `sie/open-pec-efie`，验收数据见 `docs/assessments/sie_open_pec_efie_acceptance.md`）：

- Phase 0：`mom/topology.py` 在面所在设备上用张量运算完成边分类（内部/边界/junction）、方向一致性、顶点链接和连通分量检查；`mom/open_surface.py` 提供开放网格、只在内部边上的 RWG（闭合网格上与原 `RWGBasis` 逐位一致），以及零厚度 `Box`/`Cylinder` 的结构化网格。每个 body 有类型（介质闭合 / PEC 闭合 / PEC 开放 / 阻抗薄层），`validate_scene_structure(formulation=...)` 按 formulation 判定；PEC 闭合、阻抗薄层、junction、非流形顶点和开放片与介质混合都显式拒绝。闭合 PMCHWT 在确定性 CUDA 算法下与 master 逐位一致。§10 Q1：`witwin.core` 的 `Mesh(fill_mode='surface')` 已能承载开放网格，`Box`/`Cylinder` 允许零厚度，无需改动 core。
- Phase 1：`efie_open_pec`（dense、真空），单层算子在 `mom/single_layer.py` 中把矢量位块和 P0 标量位块分开装配，可供快速后端单独作用。验收：互易性 5e-15，无损功率守恒 7e-14；方板 RCS 一阶收敛（观测阶 0.97）；圆盘静态极化率外推后与 16/3、−8/3 a³ 相差 0.03%、0.2%；带状偶极子谐振长度 0.471λ（感应电动势法 0.472λ），电抗在 5% 以内。棱边 λ/20 内近场只有约 h^0.5 收敛，单独报告。结果持久化尚未实现，`snapshot()` 显式拒绝。
- Phase 3：选用 Calderón（BC 对偶基）。开放边修正采用 “natural” 规则：边界顶点的对偶单元不带电荷，BC 通量可经单面重心函数穿过边界（非 tilde 的 H^-1/2(div) 对偶空间）；零通量规则随加密增长（27→47），natural 规则平坦（1λ 方板 8/16/24/32 格：16/18/19/20 次迭代，无预条件 56/103/145/188 次）。0.01λ–1λ 迭代 12–17 次；0.01λ 时无预条件的 1e-6 残差仍有 7.6e-4 的解误差，而 Calderón 解误差为 1.6e-6。准 Helmholtz 投影只能通过右预条件接口实现，在低频反而变差，未交付。通过 `MoMConfig(right_preconditioner=CalderonPreconditioner())` 接入 flexible GMRES，真实残差判据不变，`preconditioned_residual` 同时报告真实残差和预条件残差。

Phase 1–3 是最小可用集合：仅有 Phase 1 而没有 Phase 3，大尺寸或细网格算例无法收敛。

## 7. 验证参考

- 解析或半解析参考：PEC 球与介质球的 Mie 解；无限阻抗薄层的反射；旋转对称体（BoR）的高阶参考解（圆盘、圆柱）。
- 自洽检查：网格加密收敛、互易性、功率守恒（无损时吸收功率为零；有损时与表面损耗积分一致）、junction 电流守恒。
- 交叉验证：本仓库 FDTD 的 PCB/贴片 fixture；外部参考求解器只在 Bench 的外部参考流程中使用，遵守现有预算和授权规则。
- 指标：积分量（RCS、远场、阻抗）和近场分开报告；近场按距离棱边的远近分层报告。

## 8. 与神经求解框架的接口（InstaMax 方向）

这个计划同时为"神经网络预测表面电流，再用 Green 函数解析求场"的路线提供底层支持：

- 标签：每条 RWG（含 junction）边的电流系数，与输入 mesh 一一对应，网络可以对每个 token 直接回归。
- 奇异性剥离：边界边附近提供解析 Meixner 权重 w(r)，网络只学 J / w 的光滑部分（与 FDFD 侧减去入射场是同一个思路）。
- 物理约束与验证：提供可微、批量化的**预条件残差**算子；EFIE 的原始残差不能用作物理保证。
- 投影：在表面空间中用少量 Krylov 迭代完成，代价远低于体网格投影。
- 源：InstaMax 场景使用点偶极子，因此 Phase 0a 是这条路线的最早依赖；纯介质场景在 Phase 0a 之后即可开始原型，不需要等开放表面。
- 网络侧计划：`instamax/docs/plans/physics_constrained_generation.md`；总路线图：`witwin/neural_maxwell_solver_roadmap.md`（本计划对应其中的 S-0、S-1）。

## 9. 风险

- 开放表面的 Calderón 预条件实现复杂度高，边缘处理容易出错；如果周期过长，先交付准 Helmholtz 投影。
- 多区域 junction 的基函数和符号约定容易产生非物理电荷；junction KCL 残差必须作为每次求解的诊断输出。
- 新 formulation 会触发 Bench 的新 epoch，并需要重新运行已冻结的证据；需要提前安排。
- Meixner 基与 FMM 的多极矩接口可能不兼容；Phase 4 中先只用于 dense 和近场块。
- Known issue (measured in Phase 0b/S-1c, behaviour intentionally unchanged): the near-disjoint product rule accepts a pair when two successive orders differ by at most `near_tolerance`, and this estimate under-reports the true error slightly. Against an independent high-order reference, accepted pairs reach 1.2e-5 to 1.3e-5 for tolerance 1e-5 (coplanar and creased pairs at separation/size 1e-2). Pairs it cannot accept go to the gap-aware fallback, which stays below 4e-6. Changing the acceptance test would break bit identity with frozen evidence, so it needs its own work package and a new Bench epoch.
- Follow-up (S-1d): order-8 joint Duffy rules have a 3.3e-4 error on a well-shaped (aspect 1) shared-edge pair creased at 15 degrees, measured against the independent touching reference and order-24 Duffy. Pairs below the aspect threshold keep these rules for bit identity, so finely meshed surfaces with mild creases carry this error. A crease-angle criterion for the touching rule selection needs its own work package and a new Bench epoch.
- Follow-up (S-1d): the anisotropy-aware touching rule covers dense assembly and the full and selected-row lifted right-hand sides. Hierarchical near blocks (`nearfield.py`) still apply the joint Duffy rules to every touching pair, so H/H2/FMM operator blocks on sliver meshes keep the fixed-order touching error.

## 10. 待决问题

1. `witwin.core` 是否已支持开放三角网格几何，还是需要新增类型？
2. 厚度参数化薄导体是扩展 `LossyMetalMedium`（增加 thickness），还是新增材料类型？
3. 有限衬底（Phase 6）与分层 Green 函数（Phase 8）哪个先做，取决于目标场景（RIS 单元阵列更适合周期加分层；单个天线更适合有限衬底）。
4. 周期 kernel 与开放 EFIE 的组合（FSS/RIS 阵列）是否纳入本计划，还是归入 quasiperiodic 计划。
