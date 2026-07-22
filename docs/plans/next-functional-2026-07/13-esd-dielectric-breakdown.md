# ESD 激励、耐压评估与介电击穿开发计划

> 状态：in-progress (phases delivered 2026-07-19)  
> Delivered 2026-07-19 (Wave D selective start, owner-authorized): Phases 0–2
> (product/credibility gate, IEC 61000-4-2 standard waveform + ideal terminal
> injection, non-feedback stress/rating monitors) and Phase 4 (deterministic
> field-duration/latching dynamic dielectric breakdown with dynamic conductivity,
> typed event log, and breakdown-dissipation energy channel). Phase 3
> (electrostatic pre-bias + circuit-ESD co-simulation), Phase 5 (multi-GPU /
> scale-out / smooth surrogate), and the calibration/standard Phases 6–7 are NOT
> delivered. Evidence is E1 (analytic waveform diagnostics, golden state-machine
> and energy-closure gates, dt-convergence, bitwise no-op parity; no external
> reference solver / calibrated failure cross-check); NOT `completed`. See
> `docs/assessments/c13-esd-stress-acceptance-2026-07-19.md`,
> `docs/assessments/d13-breakdown-acceptance-2026-07-19.md`, `tests/esd/`,
> and `tests/breakdown/`.  
> **Round-H revision (2026-07-21, master `a63dee8`; merge `3f25710`).** Phase 3 circuit-driven
> ESD delivered: a new `ESDVoltageSource` drives ESD through the standard 330 Ω / 150 pF
> source-impedance network (a circuit **approximation of the standard network**, not
> discharge-gun geometry or certification), gated by an independent scipy `solve_ivp` cross-check
> sharing no runtime code (port-voltage rel 7.8e-4), a closed-box coupled energy conservation
> closure (1.34e-4), provenance ride-through, and a circuit-driven prebias+ESD end-to-end; a
> cleanup EM-load-bearing companion gate makes the field one-port load-bearing (true beats
> zeroed-EM ~12.7× on a high-impedance variant). The Phase-5 differentiable `SmoothBreakdownRisk`
> surrogate (typed **non-physical / non-regulatory**) landed with gradient/monotonicity gates.
> Phase 3 is now substantially delivered (prebias + circuit ESD) and the `TerminalPort` MNA
> coupling is tested. Evidence E1–E2 (no external / calibrated reference). **Still open:** the
> dynamic conductive breakdown feedback through the circuit-driven port is fail-closed
> (conductive-media port coefficient); Phase 5 multi-GPU / scale-out; Phases 6–7
> (surface/random/thermal feedback, gun/system calibrated-standard workflow) excluded.
> Phase-status bookkeeping / any `completed` mark is the supervisor's job (audit §4
> non-author-review bar), not set here. See
> `docs/assessments/h4-esd-circuit-acceptance-2026-07-21.md`.  
> 路线定位：分级垂直能力，当前不排期交付  
> 日期：2026-07-14  
> 目标证据：stress-only E2；deterministic breakdown E2；经校准系统预测 E3  
> 路线优先级：P3（RF 端口/电路、静电和监视器数据模型成熟后，由 EMC/ESD 产品需求触发）  
> 主要依赖：RF terminal 与 accepted power、电路/SPICE-MNA、`12-electrostatics-capacitance.md`、surface current/dissipated power monitor  
> Owner modules：`sources.py`、circuit runtime、`media.py`、`fdtd/`、`monitors.py`、`result.py`  
> 最近架构决策：2026-07-14，将应力分析、确定性击穿和经校准系统预测分层，不把阈值事件宣称为真实电弧模型  
> 可信度声明：第一阶段只做标准激励与耐压风险评估；在动态击穿/电弧模型和测量验证完成前，不宣称“预测真实 ESD 失效”

## 1. 背景与当前能力

Maxwell 已有时域 FDTD、任意电流/自定义波形、导电/色散/非线性材料、time monitor 与 CUDA 更新主干，因此可以传播快速瞬态场；但这不等价于 ESD 工程能力。gap analysis 明确指出缺少 ESD gun/标准波形、元件额定值、dielectric strength、击穿阈值 monitor、事件位置/持续时间以及 PCB 元件/网络失效映射。

ESD 项目必须拆成可验证层级：

1. **激励与应力**：复现标准电流/电压波形，报告局部 E/H、端口 V/I、表面电流和能量。
2. **耐压规则评估**：对材料/元件阈值进行可审计、非反馈的 exceedance 检测。
3. **动态介电击穿**：击穿状态改变导电率并反馈 EM，产生事件和能量沉积。
4. **系统失效/电弧**：需要电路、热、等离子体/电弧和实测校准，不能由一个阈值开关冒充。

本计划覆盖 1–3 的工程路线，并为 4 留接口；完成前按层级标注支持范围。

## 2. 目标与非目标

### 2.1 目标

- 提供 versioned ESD waveform/source，支持标准电流脉冲、接触/空气放电配置和用户测量波形。
- 提供额定电压/电流/场强/能量数据模型与类型化 exceedance 结果。
- 提供 dielectric strength、持续时间、累积损伤和恢复/锁存状态的材料击穿描述。
- 在 GPU FDTD 更新中实现局部状态转换、动态导电率与耗散能量统计。
- 将静电预偏置、ESD 瞬态、电路端口和元件/net 映射组织为可追踪工作流。
- 保持 `Scene -> Simulation -> Result`，并提供确定性的单/多 GPU 事件日志。

### 2.2 非目标

- 第一版不模拟真实火花通道的等离子体流体、随机先导、紫外光电离或几何烧蚀。
- 不凭 IEC 波形类名宣称完整标准认证；gun 几何、校准靶、布置和测量链都必须分别验证。
- 不把硬阈值击穿结果宣传为器件寿命或失效概率；统计可靠性需测量数据和概率模型。
- 不实现 PCB/CAD UI；核心提供 net/component/rating 的结构化标识和结果，Studio 负责显示。
- 不顺带启动通用 thermal/plasma multiphysics。若击穿温升需要 Bioheat/heat，只通过显式耦合接口并另行验收。

## 3. 用户工作流与 Public API 草案

### 3.1 激励与耐压评估

```python
waveform = mw.ESDWaveform.iec_61000_4_2(
    level_voltage=8_000.0,
    discharge="contact",
    standard_revision="explicit-supported-revision",
)

scene.add_port(mw.TerminalPort(
    name="gun", positive_terminal="tip", negative_terminal="return",
    reference_impedance=330.0,
))
scene.add_monitor(mw.BreakdownMonitor(
    name="dielectric_stress",
    region=connector_insulator,
    quantities=("electric_field", "exposure", "dissipated_energy"),
))

result = mw.Simulation.fdtd(
    scene=scene,
    frequencies=(1e9,),
    run_time=...,
    excitations=mw.PortExcitation(port="gun", waveform=waveform),
).run()
stress = result.breakdown("dielectric_stress")
print(stress.peak_field, stress.exceedance_duration, stress.locations)
```

### 3.2 动态击穿

```python
insulator = mw.Material(
    permittivity=3.2,
    conductivity=1e-12,
    breakdown=mw.DielectricBreakdown(
        critical_field=22e6,
        minimum_duration=2e-9,
        post_breakdown_conductivity=5e3,
        state="latching",
        model="field_duration",
    ),
)

prebias = mw.ElectrostaticInitialCondition.from_result(dc_result)
transient = mw.Simulation.fdtd(
    scene=scene,
    run_time=...,
    initial_condition=prebias,
).run()
events = transient.breakdown_events
```

建议公共对象：

- `ESDCurrentSource` / `ESDVoltageSource`：由 versioned `ESDWaveform` 与 RF terminal injection 组合；不伪装成 point dipole。
- `MeasuredWaveform(time, values, units, bandwidth, provenance)`。
- `ComponentRating(voltage=None, current=None, energy=None, pulse_width=None, model=None)`，绑定稳定 component/net id。
- `DielectricBreakdown(critical_field, model, minimum_duration, post_breakdown_conductivity, recovery, damage_parameters)`，组合进 `Material`。
- `BreakdownMonitor`、`ComponentStressMonitor`。
- `BreakdownEvent(time, position, material_id, field_before, state_before/after, deposited_energy, cause)`。
- `BreakdownResultData(events, state, stress, component_exceedance, energy_balance, provenance)`。

标准工厂函数必须要求/记录 revision；若用户省略，只能使用文档中固定且测试覆盖的默认 revision，结果写入 provenance。

## 4. 物理、离散与耦合设计

### 4.1 ESD 激励

标准电流可由双指数/多指数/Heidler 类解析函数、分段目标点拟合或测量采样给出。compiler 在 FDTD 时间网格上进行带抗混叠的积分守恒重采样，并报告：峰值电流、规定时刻电流、上升时间、总电荷 `∫I dt`、作用积分 `∫I² dt` 和能量（若端口电压可得）。

gun/source impedance 和被测设备网络属于电路耦合：理想注入仅用于 field stress 基础；生产系统 ESD 应通过 terminal + MNA/SPICE 计算端口 V/I，避免强行指定互相矛盾的电流和电压。

### 4.2 耐压与击穿准则

非反馈 monitor 支持至少：

```text
instantaneous: max_t |E(t)| / Ecrit
field-duration: integral H(|E|-Ecrit) dt and longest contiguous duration
damage: D(t) = integral g(|E|/Ecrit, T, material_state) dt
component: V(t), I(t), P(t), integral P dt relative to rating envelope
```

阈值比较使用指定位置的物理场。Yee 分量先按与能量一致的方式共点化；部分体素只对目标材料 occupancy 计入。结果报告 cell peak 和可选 conservative reconstructed peak，不能在两者间静默切换。

### 4.3 动态介电击穿

每个可击穿 material cell 维护 `intact / conducting / recovering / failed` 状态和 damage scalar。达到准则后，在明确时间层更新有效 conductivity：

```text
J = sigma(state, E, D, T) E
P_loss = J · E
```

第一生产模型是 deterministic field-duration/latching switch，用于比较和局部导通路径研究。conductivity transition 可采用有限 ramp 以满足数值稳定，ramp 时间和最大 `sigma*dt/epsilon` 受 prepare-time 稳定性检查。击穿能量进入 dissipated-power channel；若启用热耦合则作为显式 source lineage。

相邻 cell 的通道扩展、沿面闪络和随机 Weibull model 进入后续 phase。任何模型必须区分校准参数与材料常数，结果记录模型/version。

### 4.4 静电预偏置与电路耦合

- `ElectrostaticInitialCondition` 将 DC `E/D` 映射到 FDTD staggered fields，需验证离散 Gauss constraint 和边界兼容。
- 生产 terminal 路径继承 01 的 port fragment reduction 和 04 的 circuit owner/同一步 Schur 强耦合；第一版不 subcycle。任何后续 subcycling 必须作为独立 phase 推导能量合同并重新验收。
- component/net mapping 只引用稳定 id；若 geometry 改变导致映射失效，`prepare()` 报错。

## 5. 数据模型与运行时边界

prepared FDTD 增加：waveform samples/injection weights、rating monitor descriptors、breakdown material mask、state/damage tensors、model parameters 和 event buffer。基础材料 compiler 仍生成 intact EM tensors；breakdown runtime 只更新受影响系数，不每步重新 rasterize Scene。

事件日志必须是类型化数据而非 print：全局 time/step、position/global cell id、material/component/net id、触发量、阈值、持续时间、状态转换、局部能量与 model version。结果还应包含完整 source waveform diagnostics、端口 V/I、component stress envelope、全局 EM/电路/耗散能量 balance。

事件 buffer 采用 bounded/chunked 设计；溢出必须失败或按显式 aggregation policy 汇总，不能静默丢事件。

## 6. GPU-first、PyTorch 与梯度策略

- waveform resampling 后的注入、stress accumulation、state update、dynamic coefficients、event compaction 和能量 reduction 均在 GPU。
- 事件只在用户请求/运行结束时传 CPU；每步 Python callback 或 `.item()` 禁止进入生产路径。
- 未触发的固定状态区域，对材料、源幅、terminal 和连续几何参数保留现有 FDTD adjoint。
- 硬击穿事件在触发时间处不可微；真实 `BreakdownResult` 不伪造跨事件梯度。优化另提供 `SmoothBreakdownRisk`（sigmoid field margin、soft duration/damage）作为可微 surrogate，类型和文档与物理状态模型分开。
- 若需要 hybrid-event sensitivity，作为研究 phase 单独实现 saltation/event-time derivative，并必须通过有限差分；不作为初版承诺。
- trainable 场景启用 hard feedback breakdown 时，`prepare()` 默认拒绝 backward，除非所选模型明确声明支持。

## 7. Multi-GPU contract

- waveform 参数全 rank 一致；terminal injection 按端口几何 owner 分配，V/I 做 GPU collective。
- breakdown state/damage 由 cell owner 更新；邻域传播模型使用 halo，状态只由 owner 写入。
- 事件按 `(time_step, global_cell_id, event_type)` 全局排序和去重；全局 earliest/peak 使用确定性 reduction。
- component/net stress 可跨 shard，使用标量/小向量 all-reduce，不 gather 全场。
- event buffer 分 shard 保存，Result 提供逻辑统一视图；save/load 保留 shard provenance 和全局顺序。
- 单/多 GPU 必须在 source V/I、首次事件时间/位置、最终 state mask、耗散能量和无事件场上达到规定 parity。

## 8. 开发 phases、依赖与 exit gate

### Phase 0：产品与可信度 gate（E0, experimental）

Deliverables：目标标准/revision、gun/注入范围、DUT 类型、rating 数据来源、击穿声称层级和公开/实测验证件。Exit gate：标准文档法律可用、至少一套校准波形数据和材料 coupon 数据；否则保持 deferred。

### Phase 1：标准 waveform 与 terminal 注入（E1, experimental）

Deliverables：versioned ESD waveform、测量波形、重采样诊断、terminal source、V/I/Q/action integral Result。依赖：RF terminal。Exit gate：规定峰值/时刻/电荷指标达标；时间步收敛；端口电流与目标波形、source energy 对账。

### Phase 2：无反馈 stress/rating monitor（E2）

Deliverables：`BreakdownMonitor`、`ComponentRating`、field-duration/damage accumulator、位置/持续时间/区域统计。Exit gate：合成 waveform 阈值 golden tests、Yee 共点/partial voxel convergence、component V/I/P/E 逐样本 reference parity。

### Phase 3：静电预偏置与系统电路 ESD（E2）

Deliverables：DC initial condition、terminal MNA/SPICE co-simulation、component/net mapping。依赖：`12-electrostatics-capacitance.md` 和 SPICE/MNA 计划。Exit gate：DC 场映射 Gauss residual；RLC/传输线 ESD 解析/电路 reference；EM+电路能量守恒。

### Phase 4：确定性动态介电击穿（E2）

Deliverables：cell state/damage、field-duration latching model、dynamic conductivity、event log、dissipated energy。Exit gate：manufactured trigger/no-trigger、触发时间随 dt 收敛、导通路径电流和总能量闭合；无击穿时与基础 FDTD bitwise/严格 parity。

### Phase 5：multi-GPU、规模化与可微风险（E3）

Deliverables：distributed event/state、bounded buffers、smooth risk surrogate、性能调优。Exit gate：单/多 GPU event parity；surrogate gradient finite difference；启用无事件 monitor 的开销满足预算。

### Phase 6：沿面/随机/热反馈（deferred, excluded from target evidence）

相邻传播、surface flashover、Weibull hazard、温度依赖和恢复模型只有在 coupon/系统测量足以校准时启动。Exit gate 必须预注册 blind validation；否则这些保持 experimental，不能进入默认 API。

### Phase 7：gun/系统标准工作流（E3, independent certification gate）

包含实际 gun geometry、校准靶和布置。完成该 phase 才可对特定 revision 声称 workflow support；基础 waveform 支持不等于标准认证。

## 9. 验收与 benchmark 策略

- **波形/解析**：双指数/Heidler 积分、峰值、总电荷和 action integral；不同 dt 的抗混叠与收敛。
- **场传播**：TEM line/coax 注入、自由空间或屏蔽腔脉冲；端口反射与能量平衡。
- **阈值算法**：精确已知越阈区间、多个脉冲、边界恰等阈值、恢复/锁存的 state-machine golden tests。
- **介质解析/manufactured**：均匀 cell 受控 E(t) 触发；平行板介质击穿；串联介质中场分配。
- **网格/时间收敛**：峰值 E、首次触发时间、导电区域、峰值电流与耗散能量至少三档；离散事件允许报告阶梯误差带。
- **守恒**：source/circuit 输入 = 边界出流 + EM 储能变化 + 材料/击穿耗散，逐阶段检查。
- **标准/测量**：标准 calibration target 指标；材料 coupon 与至少一个板级公开/自有盲测案例。结果必须区分 calibration 和 validation 数据。
- **性能**：无击穿 monitor overhead、active breakdown overhead、event throughput、峰值显存；未触发场景目标开销不超过基础 FDTD 的 10%（最终阈值在 Phase 0 预注册）。
- **端到端**：静电预偏置→ESD terminal+电路→局部 stress/事件→component/net report→save/load，多 GPU 重现。

建议 benchmark：`esd/waveform_calibration`、`esd/terminal_rc_load`、`esd/shield_aperture`、`breakdown/uniform_cell`、`breakdown/parallel_plate`、`breakdown/layered_dielectric`、`breakdown/component_rating_board`、`breakdown/multi_gpu_event_boundary`。

## 10. 主要风险与缓解

- **过度声称预测能力**：Result 和文档标记 `stress-only / deterministic-breakdown / calibrated-system` capability level。
- **标准波形并不等于 gun**：source waveform、source impedance、gun geometry 和校准分别验收。
- **硬事件数值不稳定**：conductivity ramp、稳定性预检、dt convergence 和 energy diagnostics。
- **材料阈值高度依赖统计/环境**：参数带 provenance/温湿度/试样条件；支持 sensitivity，不提供虚假的通用默认值。
- **事件不可微**：真实 hard model 明确拒绝 backward；优化用独立 surrogate。
- **耦合范围爆炸**：电弧/等离子/热/结构损伤均需独立 product gate；核心只留接口。
- **多 GPU 非确定事件**：global id、owner write、确定性排序/reduction 固化为 contract。

## 11. 完成定义

“ESD 激励与应力分析完成”要求 Phase 1–3 的标准 waveform、terminal/circuit 注入、rating/stress 结果、静电预偏置和能量验收全部通过。“确定性介电击穿完成”还要求 Phase 4–5 的动态反馈、事件日志、时间/网格收敛、coupon/独立参考、单/多 GPU parity 和性能预算通过。只有阈值 if-statement、无电路 terminal 的 point source、或未校准的电导跳变均不计为完成；也不得据此声称真实电弧或标准认证。每个已完成层级必须同步更新 `FEATURE_LIST.md`、支持矩阵、教程、benchmark 与已知限制。
