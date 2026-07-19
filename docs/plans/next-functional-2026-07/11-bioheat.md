# 生物热与 EM 温升耦合开发计划

> 状态：proposed  
> 路线定位：垂直行业后续项目，当前不排期交付  
> 日期：2026-07-14  
> 目标证据：基础 Bioheat E3；温度反馈 EM 为条件 E2 gate  
> 路线优先级：P3（共享耗散功率数据成熟后，由生物电磁产品需求触发）  
> 主要依赖：`01-rf-engineering-workflow.md` Phase 4 的 `PowerLossData`（可复用 `10-sar.md` Phase 1 实现）、热材料/边界数据、可复用 GPU 标量 PDE 基础设施  
> Owner modules：拟新增 thermal/bioheat 模块、shared compiler、`simulation.py`、`result.py`  
> 最近架构决策：2026-07-14，默认提供守恒的一向 EM→Bioheat 耦合，双向温度反馈单独通过产品 gate  
> 边界声明：EME 和未在总路线列出的其他多物理求解器仍暂缓；本文只规划已明确列出的 Bioheat 项目

## 1. 背景与当前能力

Maxwell 当前是全波 EM 框架，具有声明式 `Scene`、GPU FDTD、材料编译、torch 结果和 EM 损耗相关基础数据，但没有热传导或 Pennes bioheat 求解器。两份 gap analysis 都把热传导/生物热判定为独立 PDE 产品线：现有 `Result` 可以成为耦合数据载体，不能把温升实现成对 SAR 乘一个经验常数。

本计划面向人体暴露、植入物、RF 消融和长期温升评估。路线优先级低于 RF 工程闭环和 SAR；只有当目标市场需要温升/HAC/MRI 或已有 SAR 客户明确要求时才启动。计划先交付可验证的单向 EM→thermal，再考虑温度反馈 EM 材料；不以“大而全多物理平台”为目标。

## 2. 目标与非目标

### 2.1 目标

- 在相同公共 `Scene` 几何/网格语义上求解稳态和瞬态 Pennes bioheat equation。
- 支持热导率、密度、比热、灌注、血液温度、代谢热和体热源。
- 支持定温、热流、对流和绝热边界，以及材料界面热流连续。
- 从 EM 耗散功率密度或 SAR 构造守恒的 thermal source，并保留映射 provenance。
- 输出温度、温升、热流、能量平衡、区域统计与时间曲线。
- GPU-first、PyTorch-native，并为材料/源参数和连续几何参数提供梯度路径。

### 2.2 非目标

- 第一版不实现流体动力学、血管网络、相变、组织变形、热辐射视因子或化学损伤全模型。
- 不在第一版实现电磁场与温度的全隐式单体求解；先做显式声明的单向和 staggered fixed-point 耦合。
- 不自动宣称 HAC、MRI 或临床安全认证；标准工作流必须有单独产品 gate。
- 不建立与 `Scene` 并行的 `ThermalScene` 公共世界；专用内部 mesh 可以存在，但必须从同一 Scene 编译。

## 3. 用户工作流与 Public API 草案

热属性使用组合描述，避免把所有 EM `Material` 构造参数无限摊平：

```python
tissue = mw.Material(
    permittivity=41.4,
    conductivity=0.87,
    mass_density=1050.0,
    thermal=mw.BioheatMaterial(
        thermal_conductivity=0.52,  # W / (m K)
        specific_heat=3600.0,       # J / (kg K)
        perfusion_rate=0.006,        # 1 / s, API 固定定义
        metabolic_heat=420.0,        # W / m^3
    ),
)

em_result = mw.Simulation.fdtd(scene=scene, frequencies=(2.45e9,)).run()
heat_source = mw.EMHeatSource.from_result(
    em_result,
    monitor="absorbed_power",
    normalization=mw.PowerNormalization.accepted_power("feed", watts=1.0),
)

thermal = mw.Simulation.bioheat(
    scene=scene,
    sources=(heat_source,),
    initial_temperature=310.15,
    run_time=mw.ThermalTimeConfig(duration=600.0, sample_interval=1.0),
    boundary=mw.ThermalBoundarySpec.convection(h=8.0, ambient_temperature=298.15),
).run()

delta_t = thermal.temperature.rise
```

建议公共对象：`BioheatMaterial`、`ThermalBoundarySpec`、`ThermalTimeConfig`、`HeatSource`、`EMHeatSource`、`TemperatureMonitor`、`ThermalFluxMonitor`。`Simulation.bioheat(...)` 扩展 `SimulationMethod`，返回标准 `Result(method="bioheat")`；结果通过类型化 accessor 提供 `temperature`, `heat_flux`, `thermal_energy_balance`，而非将温度塞入 `fields["EX"]` 一类命名空间。

耦合工作流的第二阶段可提供：

```python
coupled = mw.Simulation.em_bioheat(
    em=em_sim,
    thermal=thermal_spec,
    coupling=mw.StaggeredCoupling(interval=10.0, tolerance=1e-3, max_iterations=20),
)
result = coupled.run()  # 仍返回 Result；内含有版本的子阶段 lineage
```

该 classmethod 构造的仍是标准 `Simulation`；耦合编排作为内部 runtime plan，不引入 `CoupledSimulation`、第二套场景或第二套结果合同。

## 4. 物理方程、离散与耦合设计

### 4.1 Pennes 方程

```text
rho c dT/dt = div(k grad T)
              + rho_b c_b w_b (T_b - T)
              + Q_met + Q_em + Q_external
```

其中 `w_b` 的 API 定义必须固定为 1/s；若导入数据采用 kg/(m³ s) 等 convention，由 adapter 显式换算。稳态去掉时间项。各向异性热导允许 3×3 对称正定张量；第一版灌注为 cell-local 线性 sink/source。

边界：Dirichlet `T=T0`、Neumann `-n·k∇T=q`、Robin `-n·k∇T=h(T-T_inf)`、adiabatic `q=0`。材料界面满足温度与法向热流连续；可选 contact resistance 放在后续 phase。

### 4.2 空间离散

- 优先采用 cell-centered finite volume，与 SAR/材料 occupancy/cell volume 对齐。
- face thermal conductivity 使用守恒 harmonic averaging；非均匀笛卡尔网格使用真实 face area 与 center distance。
- cut/partial cells 复用 geometry occupancy，但为避免极小 cell 刚性需合并/稳定化策略及 minimum-volume diagnostics。
- 第一版复用 Scene 网格；独立 thermal coarsening 仅在保守 EM→thermal remap 已验证后开放。

### 4.3 时间推进与线性求解

- 默认 backward Euler，提供 Crank–Nicolson；显式推进只用于 reference/test，不作为生产默认。
- 每步求解 `(M/dt + K + P) T[n+1] = rhs`；GPU sparse operator 采用 matrix-free stencil 或 torch-compatible CSR，预条件 Krylov 为主。
- 自动步长依据用户误差目标和热时间常数，不沿用 FDTD CFL 时间步。
- steady-state 直接求解椭圆系统并报告残差和能量不平衡。

### 4.4 EM→thermal 映射

EM source 只消费共享 `PowerLossData` 的 volume total channel，`Q_em [W/m3]` 保留 conduction/dispersion/magnetic 等 lineage。surface W/m2 与 wire W/m 必须通过守恒的 measure-to-volume remap 进入 thermal cells，并满足积分功率不变；不能直接当 W/m3 使用。同网格 volume channel 复用 cell ownership；不同 thermal grid 使用保体积积分 remap，必须满足 `sum(Q_em V)` 在容差内守恒。频域源通常为周期平均；脉冲源使用时间 bin 能量，不允许在未声明 averaging window 时混合。

双向耦合时，温度通过显式 `MaterialPerturbation` 更新介电率、导电率等，执行：EM solve → loss remap → thermal advance → material update → convergence check。每轮记录能量、最大温变和材料变化，发散时失败而非返回最后一次结果冒充收敛。

## 5. 数据模型与运行时边界

prepared scene 增加独立的 thermal block：`rho`, `c`, `k`, `perfusion`, `blood_temperature`, `Q_met`, `occupancy`, boundary faces。EM material tensors 不依赖 thermal runtime，但二者共享几何/material id/hash。

`ThermalResultData` 至少包含：

- `temperature[T, x, y, z]` 或请求的采样时刻；
- `temperature_rise`、`heat_flux`、`source_power_density`；
- boundary/input/perfusion/storage 能量分解与 balance residual；
- time coordinates、SI units、solver residual history；
- source lineage：EM result hash、频率、功率 normalization、remap error；
- material/boundary/grid hashes。

默认不保存所有时步全场；monitor 决定采样，最终场与标量诊断始终可用。

## 6. GPU-first、PyTorch 与梯度策略

- 材料编译、FVM operator、线性迭代、source remap 和 monitor reduction 均在 GPU；CPU 只负责配置、文件解析和小 metadata。
- 稳态/隐式时间步通过 custom autograd implicit differentiation，反向求解转置线性系统，避免存储全部 Krylov 历史。
- 对多步瞬态使用 checkpoint/recompute 或 discrete adjoint；用户可配置 checkpoint stride。
- 支持对 `k, c, rho, perfusion, Q_met, Q_em` 和连续 occupancy 求梯度；硬材料 id 和自适应步长分支默认不可微。
- 结果的温度统计与固定时间/区域目标保持 torch 链；peak time/location 提供真实离散值和单独的 smooth surrogate。
- 若某 solver/preconditioner 无伴随实现，trainable 场景必须在 `prepare()` 阶段明确拒绝，不能静默 detach。

## 7. Multi-GPU contract

- cell-centered temperature、material block 和 source 按 thermal domain shard；face flux 使用相邻 halo。
- 隐式 Krylov 的 dot/norm/residual 通过 GPU all-reduce；收敛判据与单 GPU 使用同一全局范数定义。
- EM 和 thermal 使用同一 partition 时零拷贝映射；不同 partition 使用 device-to-device conservative redistribution。
- boundary energy、perfusion sink、storage term 和区域统计做确定性归约；结果报告 global balance。
- 单次大问题采用域分解；参数/功率/姿态扫描采用任务级多 GPU。不得为多 GPU 增加 CPU gather 全场步骤。
- 反向使用相同 ownership/halo 逆通信合同，并以单 GPU gradient parity 为 gate。

## 8. 开发 phases、依赖与 exit gate

### Phase 0：产品触发与方程合同（E0, experimental）

Deliverables：目标用例、材料单位、边界约定、Pennes 参数定义、验证数据许可、性能尺寸。Exit gate：至少一个真实客户/产品用例和一个公开独立参考；否则项目保持 deferred。

### Phase 1：稳态热传导核心（E1, experimental）

Deliverables：`BioheatMaterial` 基础热参数、FVM operator、Dirichlet/Neumann/Robin、steady solver、温度/热流 Result。Exit gate：1D slab、复合墙、带对流 sphere/cylinder 的解析误差与三档网格收敛达标；全局热平衡闭合。

### Phase 2：瞬态与灌注/代谢（E2）

Deliverables：backward Euler/CN、Pennes perfusion、时间 monitor、自动步长、checkpoint。Exit gate：lumped capacitance、瞬态 slab、manufactured solution 和 time-step convergence；长时间趋近 steady 解。

### Phase 3：单向 EM→Bioheat（E2）

Deliverables：`EMHeatSource.from_result`、volume/surface/wire 到 thermal volume 的守恒 remap、功率 normalization lineage、耦合 Result。依赖：共享 `PowerLossData`（可选复用 SAR Phase 1）。Exit gate：各 measure 映射总功率守恒；均匀加热解析温升；天线/phantom 端到端与独立参考一致。

### Phase 4：可微与 multi-GPU（E3）

Deliverables：implicit backward、transient adjoint、domain decomposition、性能调优。Exit gate：有限差分梯度；单/多 GPU field/residual/gradient parity；目标规模不发生 host gather。

### Phase 5：温度反馈 EM（E2, conditional）

Deliverables：温度依赖材料 perturbation、staggered coupling、松弛与收敛控制。Exit gate：manufactured coupled case、能量/迭代收敛、不同 coupling interval 收敛；未收敛路径可靠失败。

### Phase 6：行业扩展（deferred, excluded from target evidence）

HAC、MRI 指标、thermal dose/Arrhenius damage、显式血管网络均在此评审，不因基础 bioheat 完成自动列为支持。

## 9. 验收与 benchmark 策略

- **解析/manufactured**：定常线性温度梯度、均匀体热源 slab、对流边界、分层导热、带灌注指数响应。
- **网格/时间收敛**：至少三档空间网格和三档时间步，报告 L2/Linf、热流和总能量误差；CN/BE 各自达到预期阶。
- **守恒**：输入源 + 代谢 + 灌注交换 = 边界散热 + 储能，逐时步和全程积分均检查。
- **标准/独立参考**：公开 Pennes benchmark 或文献可复现实验；固定方程 convention 和参数来源。
- **性能**：cells·steps/s、Krylov iterations、峰值显存、remap cost；大场景热求解不得因保存全时序而线性爆内存。
- **端到端**：RF source→accepted power→absorbed density→10 min temperature rise→区域/峰值统计→save/load。
- **梯度**：稳态和短瞬态对 k、perfusion、source amplitude 的 central difference/complex-step（适用时）验证。

建议 benchmark：`thermal/steady_slab`、`thermal/composite_wall`、`thermal/convection_sphere`、`bioheat/perfused_cube`、`bioheat/transient_manufactured`、`bioheat/em_heated_phantom`、`bioheat/temperature_feedback_slab`。

## 10. 主要风险与缓解

- **参数 convention 混乱**：所有字段使用 SI，perfusion 在 API 类型/文档中固定定义，adapter 负责换算。
- **几何网格不适合热扩散**：先同网格保证正确；coarsening 必须过 conservative remap gate。
- **极小 cut cell 条件数恶化**：诊断、cell merge/稳定化与预条件器作为 Phase 1 必需项。
- **双向耦合不收敛**：默认单向；双向要求松弛、残差和 max iterations，失败可诊断。
- **梯度内存过高**：隐式反向和 checkpointing，不展开 Krylov/全时间图。
- **路线膨胀**：临床损伤、HAC/MRI、CFD 都需独立产品 gate，不能偷渡进核心 phase。

## 11. 完成定义

基础 Bioheat 完成要求：稳态与瞬态 Pennes、四类边界、EM 单向耦合、类型化 Result、SI/provenance、热平衡和收敛验证全部可用；公开 benchmark 与独立参考达标；关键参数梯度和单/多 GPU parity 通过；规模性能满足预注册预算；教程完整展示 `Scene -> EM Result -> Bioheat Simulation -> Result`。只有壳类、CPU reference 或无守恒 remap 的温升图不计为完成。温度反馈 EM、HAC/MRI 等只在各自 phase exit gate 后单独标记支持；每个用户可见层级必须在同一变更中更新 `FEATURE_LIST.md`、支持矩阵和 known limitations。
