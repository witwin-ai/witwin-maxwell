# 静电与电容矩阵开发计划

> 状态：in-progress (phases delivered 2026-07-19)  
> Delivered 2026-07-19 (Wave D selective start, owner-authorized): Phases 0–3
> (product/convention freeze, scalar Laplace/Poisson matrix-free FVM + PCG,
> floating conductors + gauge handling, N-terminal Maxwell capacitance matrix)
> plus a Phase-5 differentiability slice (implicit-differentiation backward,
> central-difference gradient gates). Phase 4 (nonuniform grid / tensor eps /
> open boundary) and the Phase-5 multi-GPU/large-scale piece remain fail-closed.
> Evidence is E1–E2 (analytic / convergence / conservation / energy-identity /
> gradient gates, no external reference solver cross-check); NOT `completed`.
> See `docs/assessments/a12-electrostatics-acceptance-2026-07-19.md` and
> `tests/electrostatic/`.  
> **Round-H revision (2026-07-21, master `a63dee8`; merge `aa02075`).** Phase 4 SPD
> tensor-eps + open boundary delivered: a full SPD 3×3 tensor permittivity in the FVM operator
> (`A = A_diag + A_cross`, `A_cross` = gradient of a discrete quadratic energy → symmetric by
> construction), gated by dense/random operator symmetry, rotated-frame MMS 2nd-order
> convergence, anisotropic-capacitance reciprocity (<1e-6) and energy identity; an `open`
> boundary now fails closed; and the opt-in `truncation_estimate` domain-extension API reports a
> base/enlarged/delta + 1/L Richardson infinite-domain limit. Cleanup added a wall-tangential
> MMS (exposing the documented 1st-order wall cross-flux), a precomputed-stencil `_apply_cross`
> (bit-equal, no per-iteration autograd), and a boundary-touching-structure confound fail-close.
> Trainable tensor-eps backward stays fail-closed (Phase 5). Evidence E2-class for the delivered
> envelope (no external reference). **Still open:** exact (BEM) open boundary, trainable
> tensor-eps backward (P5), 2nd-order wall cross-flux, multi-GPU (P5), touchscreen workflow (P6).
> Phase-status bookkeeping / any `completed` mark is the supervisor's job (audit §4
> non-author-review bar), not set here. See
> `docs/assessments/h2-es-tensor-acceptance-2026-07-21.md`.  
> 路线定位：独立求解器后续项目，当前不排期交付  
> 日期：2026-07-14  
> 目标证据：E3 production  
> 路线优先级：P3（RF 闭环与 multi-GPU 基础设施之后，触控/封装/ESD 产品需求触发）  
> 主要依赖：共享 Scene 几何与材料编译、GPU 稀疏/矩阵自由线性求解、端子命名合同  
> Owner modules：拟新增 electrostatic/capacitance solver、shared compiler、`simulation.py`、`result.py`  
> 最近架构决策：2026-07-14，使用独立 Laplace/Poisson method 与导体约束，不以低频全波近似静电  
> 边界声明：这是总路线明确列出的静电项目；不顺带启动 EME、磁静态、导电或其他未列出的多物理求解器

## 1. 背景与当前能力

Maxwell 当前的 FDTD 求解时域全波问题，不能用趋近零频率的全波近似可靠替代 Laplace/Poisson electrostatics。现有 `Scene` 已提供三维 domain、结构、材料、网格、边界和 GPU device，`Simulation`/`Result` 是稳定公共架构；但没有导体等势/总电荷约束、电势结果、电荷密度、静电能量或 Maxwell capacitance matrix。

该项目服务封装互连、传感器、touchscreen、寄生电容和 ESD 场强前置分析。它是独立 PDE/runtime，不是给全波 solver 增加近零频率特殊分支。第一版复用结构化网格和材料 compiler；若未来需要非结构体 FEM，应另立基础设施计划，不能在本项目中建立第二套公共 Scene。

## 2. 目标与非目标

### 2.1 目标

- 求解各向同性/各向异性介质中的 3D Laplace/Poisson 方程。
- 支持定电势导体、指定总电荷的浮置导体、接地导体和绝缘/对称边界。
- 输出电势、E 场、位移场 D、体/表面电荷、导体净电荷、场能和残差。
- 自动提取 N 端子的 Maxwell capacitance matrix，并派生 mutual/self/两端等效电容。
- 保持 `Scene -> Simulation -> Result`，材料/几何参数可为 torch tensor。
- 支持单 GPU、multi-GPU domain decomposition 和批量端子激励。

### 2.2 非目标

- 不用静电求解器替代稳态导电、magnetostatics、电感提取或 full-wave RF。
- 第一版不实现半导体 Poisson–drift-diffusion、量子修正、电双层或等离子体。
- 不实现 CAD/PCB/触屏 GUI；Studio 可消费结构化结果，但后端负责真实电容物理。
- 第一版不承诺无限域 boundary element method；开放边界通过受控截断/扩域验证，精确开放边界作为后续 phase。
- ESD 动态击穿属于 `13-esd-dielectric-breakdown.md`；本文只提供静态场强、能量和电容基线。

## 3. 用户工作流与 Public API 草案

```python
import witwin.maxwell as mw

scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
scene.add_structure(mw.Structure(geometry=dielectric, material=substrate))
scene.add_electrostatic_terminal(mw.ElectrostaticTerminal(name="tx", geometry=tx, potential=1.0))
scene.add_electrostatic_terminal(mw.ElectrostaticTerminal(name="rx", geometry=rx, potential=0.0))
scene.add_electrostatic_terminal(mw.ElectrostaticTerminal(name="shield", geometry=shield, grounded=True))

result = mw.Simulation.electrostatic(
    scene=scene,
    boundary=mw.ElectrostaticBoundarySpec.grounded_box(),
    solver=mw.ElectrostaticSolverConfig(tolerance=1e-9),
).run()

potential = result.electrostatic.potential
charge_tx = result.electrostatic.terminal_charge("tx")

matrix_result = mw.Simulation.capacitance(
    scene=scene,
    terminals=("tx", "rx", "shield"),
    reference="shield",
).run()
C = matrix_result.capacitance.matrix
```

建议对象：

- `ElectrostaticTerminal(name, geometry|structure, potential=None, charge=None, grounded=False)`；potential/charge 互斥，浮置端子可指定总电荷。
- `ChargeDensity(geometry, density)` 作为体自由电荷源，单位 C/m³；后续可扩展 surface charge source。
- `ElectrostaticBoundarySpec.dirichlet/neumann/grounded_box/symmetry`。
- `Simulation.electrostatic(...)` 返回标准 `Result(method="electrostatic")`。
- `Scene.add_electrostatic_terminal(...)` 使用 solver-specific collection；它不进入 RF `Scene.ports`，避免 RF port compiler 把等势约束解释为 Z0/V/I/power-wave 端口。
- `Simulation.capacitance(...)` 是标准 `Simulation` 的多激励配置，内部复用 electrostatic runtime，结果为标准 `Result.capacitance: CapacitanceData`，不复用 RF `NetworkData`，也不另建 solver 入口模型。
- `ElectrostaticResultData` 与 `CapacitanceData`，后者提供 `matrix`, `terminal_order`, `reference`, `charges`, `energy`, `reciprocity_error`, `row_sum_error`。

端子公共语义与 RF `TerminalPort` 只共用不可变 `TerminalGeometry/TerminalRef`（名称、导体选择、正方向）基础描述；collection、compiler capability 和 Result 类型严格分开。静电 terminal 表示等势约束，RF port 表示激励/测量路径。

## 4. 物理方程、离散与约束设计

### 4.1 控制方程

```text
E = -grad(phi)
D = epsilon · E
div(D) = rho_free
=> -div(epsilon · grad(phi)) = rho_free
```

介质 `epsilon` 可为空间变化标量、对角或对称正定 3×3 tensor。第一版不接受频散复介电率；静电参数必须为实数正定的 DC permittivity。若材料只提供频域色散，用户必须显式给出 `static_permittivity`，不能猜测零频极限。

静电能量：`W = 0.5 ∫ E·D dV = 0.5 Σ V_i Q_i`。导体自由电荷由包围导体边界的 `D·n` 通量积分得到；体自由电荷纳入全局 Gauss balance。

### 4.2 导体约束与 gauge

- 定电势/接地导体：mask 内所有自由度等势并施加 Dirichlet 值。
- 浮置导体：电势为未知标量，同时满足指定净电荷；采用约束消元或 Lagrange multiplier saddle system。
- 纯 Neumann 问题需满足总电荷兼容性并固定 gauge（如体积平均 phi=0）；不满足时 prepare 阶段失败。
- 重叠端子、未连接的同名导体、被网格吞掉的薄导体和无参考的 capacitance extraction 必须诊断。

### 4.3 空间离散

- 标量/对角 epsilon 首版采用守恒 finite volume：potential 位于 cell/node 统一约定的位置，face D 使用介质界面 harmonic flux，满足离散 Gauss law。
- 全张量 epsilon 需要包含 cross-derivative 的对称离散；operator 必须保持对称正定（无浮置约束时），否则不得声称支持。
- 非均匀 Cartesian 网格使用真实 face area、cell volume 和中心距离。
- conductor boundary 的 subpixel/cut-cell flux 与现有 occupancy 对齐；窄间隙自动给出 cells-across-gap 诊断，不能用数值稳定掩盖几何未解析。
- matrix-free stencil 为生产路径；小问题 CSR reference 用于 parity 和调试。

### 4.4 电容矩阵提取

对 N 个端子依次施加 `V_j=1`、其余参考端子为 0，测得 `Q_i`：`C_ij = Q_i / V_j`。完整 Maxwell matrix 应对称、对角非负、非对角非正（在标准无源介质 convention 下）；包含无限/外壳参考时行和性质按 boundary convention 报告。

多 RHS 应批量或 block Krylov 求解，并复用 operator/preconditioner。结果保留 raw matrix 和经用户显式请求的 symmetry projection；默认不得静默用 `(C+C.T)/2` 掩盖误差。两端电容、partial capacitance 等派生量必须注明 convention。

## 5. 数据模型与编译边界

prepared electrostatic block：`epsilon_static`, `free_charge`, `cell_volume`, `face_area`, `conductor_id`, `terminal_constraints`, `occupancy`, `boundary_conditions`。它可复用 Scene rasterization/material id，但不复用 FDTD staggered update tensors。

结果坐标和单位固定：phi [V]、E [V/m]、D [C/m²]、rho [C/m³]、surface charge [C/m²]、terminal charge [C]、energy [J]、C [F]。每项包含 grid/material/terminal hash、solver tolerance、residual、Gauss error 和 boundary truncation metadata。

核心 capacitance 数据不能放入匿名 `metadata`; `Result.capacitance` 返回类型化 accessor，save/load 保持 torch dtype/device 信息和端子顺序。

## 6. GPU-first、PyTorch 与梯度策略

- raster 后的 operator apply、Krylov、preconditioner、charge flux、energy 和 matrix RHS 全在 GPU。
- 小规模 CPU direct solver 只允许测试 reference，不能成为用户 fallback 或功能支持条件。
- 对线性系统 `A(theta) phi=b(theta)` 使用 implicit differentiation；反向求解 `A^T lambda=dL/dphi`。SPD 路径可复用 forward preconditioner。
- 支持对 epsilon、charge density、terminal voltage 和连续 geometry occupancy 求梯度；hard conductor topology/terminal membership 默认不可微。
- capacitance matrix 对材料/连续几何可微；端子激励批次反向应复用 block solve，避免 N 倍 Python graph。
- trainable 场景若启用不支持 backward 的 tensor anisotropy、cut-cell 或 preconditioner，必须在 `prepare()` 拒绝。

## 7. Multi-GPU contract

- potential/epsilon/free charge 按域分片；stencil halo 与 face ownership 由全局 partition contract 决定。
- 导体可能跨 shard：terminal equipotential scalar 和净电荷约束通过小规模 device collective 归约，不 gather 导体表面全数据。
- Krylov dot/norm、gauge、Gauss balance 和 terminal charge 用 GPU all-reduce；确定性模式固定 reduction tree/accumulation dtype。
- N 个 capacitance RHS 可在单次域分解中 block-parallel，也可任务级分配端子激励；两种模式必须产生一致端子顺序和矩阵。
- surface charge face 只由 face owner 积分，避免 shard 边界双计数。
- backward 使用相同 halo/collective 反向合同，并以 field、charge、C 和 gradient parity 为验收项。

## 8. 开发 phases、依赖与 exit gate

### Phase 0：产品 gate 与 convention 冻结（E0, experimental）

Deliverables：目标 touch/package/ESD use case、potential grid convention、terminal/reference、static epsilon、C matrix 符号和 boundary truncation 规范。Exit gate：API/数值设计评审和至少一套公开 reference 数据；未满足则保持 deferred。

### Phase 1：标量介质 Laplace/Poisson（E1, experimental）

Deliverables：GPU FVM operator、Dirichlet/Neumann、体自由电荷、phi/E/D/energy Result、residual/Gauss diagnostics。Exit gate：平行板、同心球、同轴柱解析解；三档网格收敛；离散 Gauss/能量闭合。

### Phase 2：导体与浮置约束（E2）

Deliverables：terminal compiler、接地/定势/定电荷浮置导体、gauge 与错误诊断、surface charge。Exit gate：孤立/浮置导体、charge conservation、不同 gauge 下 E/D/Q 不变；cut-cell 导体收敛。

### Phase 3：N 端电容矩阵（E2）

Deliverables：多 RHS/block solve、`CapacitanceData`、self/mutual/equivalent 派生量、batch execution。Exit gate：矩阵对称性、能量等价 `0.5 V^T C V`、行和/符号性质和独立 solver 对照达标。

### Phase 4：非均匀网格、张量介质与开放边界（E2）

Deliverables：非均匀几何量、SPD tensor epsilon、可控的开放边界/扩域 convergence。Exit gate：旋转各向异性 manufactured solution；domain enlargement convergence；operator symmetry/positive-energy test。

### Phase 5：可微与 multi-GPU（E3）

Deliverables：implicit backward、distributed stencil/constraints/block RHS、性能优化。Exit gate：有限差分梯度、单/多 GPU field/Q/C/gradient parity、大规模无 host gather。

### Phase 6：touchscreen/封装工作流（E3, conditional）

Deliverables：端子阵列 sweep、差分/邻接电容统计、制造容差优化示例。Exit gate：公开触控/封装结构端到端与测量或独立 solver 对照；不将 UI/CAD 导入算作 solver 完成。

## 9. 验收与 benchmark 策略

- **解析**：平行板（含边缘效应域扩展研究）、同心球、同轴圆柱、介质分层 capacitor、点/均匀 charge Poisson manufactured case。
- **离散**：face flux、material interface、cut cell、薄间隙、非均匀网格、浮置 constraint 的 operator-level tests。
- **收敛**：空间至少三档；开放边界同时做网格和 domain-size 两轴收敛。
- **守恒**：`∮D·n = Q_free`、所有导体/体电荷总和、`0.5∫E·D = 0.5ΣVQ`。
- **矩阵性质**：reciprocity、对称性、positive semidefinite/definite（视参考 convention）、端子重排 invariance。
- **独立参考**：至少一个公开 FEM/BEM 或测量案例，固定版本、网格和边界；不只比较可视图。
- **性能**：operator GB/s/cells/s、iterations、multi-RHS speedup、峰值显存和 setup amortization。
- **端到端**：三端触控/封装结构，从 Scene 到 C matrix、参数梯度、save/load 和多 GPU parity。

建议 benchmark：`electrostatic/parallel_plate`、`coaxial_capacitor`、`concentric_spheres`、`layered_dielectric`、`floating_conductor`、`anisotropic_cube`、`three_terminal_matrix`、`touch_sensor_cell`。

## 10. 主要风险与缓解

- **把全波低频近似当静电**：独立 method/runtime，禁止内部调用极低频全波仿真冒充。
- **导体边界不解析**：mesh diagnostics、subpixel/cut-cell 和网格收敛是 exit gate。
- **开放边界误差污染 C**：报告 domain truncation，要求扩域 convergence，后续再加入专用 open boundary。
- **矩阵 convention 混乱**：类型化 `CapacitanceData` 固定 Maxwell matrix 定义，派生量显式命名。
- **浮置约束导致病态**：gauge/兼容性预检、scaling 和 block preconditioner。
- **范围蔓延到半导体/磁静态**：这些均不在本文 phases；需要独立立项。

## 11. 完成定义

项目完成必须同时具备：静电 Laplace/Poisson、导体/浮置约束、类型化 field/charge/energy Result、N 端 C matrix；解析与独立参考、三档网格/domain 收敛、Gauss/能量/reciprocity 验收通过；关键参数梯度与单/多 GPU parity 通过；规模性能满足预算；API、教程、`FEATURE_LIST.md` 和 benchmark 结果同步。仅有 Poisson demo、CPU 稀疏求解或无导体电荷守恒的 C 数字，不计为完成。
