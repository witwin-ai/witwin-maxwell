# 非线性电路器件开发计划

状态：proposed  
日期：2026-07-14  
目标证据：E3 production（核心器件）；Phase 5 transistor 为独立 E2 gate  
依赖：`04-spice-mna-cosimulation.md` 完成线性 MNA、场-电路强耦合、checkpoint 与 owner contract  
Owner modules：`circuits.py`/`circuit_devices.py`（拟新增）、`compiler/`、`fdtd/`、`result.py`  
最近架构决策：2026-07-14，先交付 charge-consistent diode/behavioral devices，晶体管模型独立 go/no-go

## 1. 背景与当前能力

Maxwell 已有非线性电磁材料，但材料本构非线性不能替代端子器件。`04` 将建立线性 MNA；若没有非线性 residual/Jacobian、Newton 收敛控制和电荷型器件历史，仍无法模拟整流、限幅、开关、功放负载牵引或电压相关电容。

本计划在原生 GPU MNA 上增加受控、分阶段的非线性器件族。第一目标是可验证且可微的 diode/非线性 I-V/Q-V，不以“能解析任意 SPICE model card”作为成熟标志。MOS/BJT 只有在基础 Newton、能量、梯度和长时间稳定性 gate 通过后进入生产范围。

## 2. 目标与非目标

### 2.1 目标

1. 定义器件 residual/current/charge/Jacobian contract，支持静态 I-V 和动态 Q-V 非线性。
2. 实现 GPU Newton-Raphson、damping/line search、limiting、breakpoint 与失败诊断。
3. 首批交付 Diode、PiecewiseLinearIV、PolynomialIV、VoltageDependentCapacitor；随后评估 BJT 和基础 MOSFET compact model。
4. 与 FDTD port 强耦合，在同一个 time step 内收敛场-电路未知量，保证无隐式一拍延迟。
5. 支持器件参数 PyTorch 梯度和离散伴随/隐函数 backward，不保存全部 Newton graph。
6. 沿用 `04` circuit owner，多 GPU 只通信端口量，不分布式拆 Newton system。

### 2.2 非目标

- 第一版不宣称完整 SPICE BSIM/PSP/HiSIM、foundry model card 或 Verilog-A 兼容。
- 不实现 RF harmonic balance、PSS/PAC、noise、温度/自热和随机 mismatch。
- 不支持任意 Python callable 在每个 CUDA time step 中求 I-V；用户自定义模型必须是可编译 tensor expression 或后续受控 device protocol。
- 不为收敛而 silent clamp 到错误解、转 CPU、跳过 time step 或自动改变全局物理参数。
- 不在此计划实现数字逻辑事件模拟或大型晶体管级 IC；目标是 EM-connected 小型非线性网络。

## 3. 用户功能描述与 Public API 草案

```python
import torch
import witwin.maxwell as mw

circuit = mw.Circuit(name="rectifier")
vin, vout, gnd = circuit.node("in"), circuit.node("out"), circuit.ground
circuit.add(mw.Diode(
    "d1", vin, vout,
    saturation_current=torch.tensor(1e-12, device="cuda"),
    ideality=1.05,
    series_resistance=0.5,
    junction_capacitance=0.2e-12,
))
circuit.add(mw.Resistor("load", vout, gnd, resistance=1e3))
circuit.bind_port("antenna_feed", positive=vin, negative=gnd)
scene.add_circuit(circuit)

result = mw.Simulation.fdtd(
    scene,
    frequencies=freqs,
    config=mw.FDTDConfig(
        circuit=mw.NonlinearSolveConfig(
            relative_tolerance=1e-7,
            absolute_tolerance=1e-10,
            max_iterations=20,
            line_search="backtracking",
            failure="raise",
        )
    ),
).run()

d = result.circuit("rectifier").device("d1")
loss = -d.average_delivered_power
loss.backward()
```

可控自定义曲线先提供无 Python runtime callback 的参数化对象：

```python
mw.PiecewiseLinearIV("protector", p, n, voltages=v, currents=i)
mw.PolynomialIV("behavioral", p, n, coefficients=coeff)
mw.VoltageDependentCapacitor("varactor", p, n, q_coefficients=q_coeff)
```

公共契约：

- 非线性器件只能加入 `Circuit`，再经 `Scene.add_circuit(...)` 进入标准仿真入口。
- `Scene.compile_circuits(...)` 调用 `compile_nonlinear_devices` 生成定形 residual/Jacobian plan；命名遵守 `compile_*`。
- `Result.circuit(name)` 增加每步/汇总 Newton stats；`DeviceData` 提供端压、支流、charge、瞬时/平均功率和 model diagnostics。
- `failure="raise"` 是生产默认；可选 `record_and_stop` 用于诊断，不提供继续使用未收敛值的模式。

## 4. 器件与求解数据模型

### 4.1 器件 contract

内部 `CompiledNonlinearDevice` 对局部 terminal voltage `v`、history `h` 和参数 `p` 提供：

- conduction current `i(v,p,t)` 与 `di/dv`；
- stored charge `q(v,p,t)` 与 `dq/dv`；
- 可选 branch residual/Jacobian stamp；
- voltage/current limiting state；
- power `p=vi`、stored energy（可定义时）和参数域 validation。

所有输出为定形 CUDA tensors；同类型器件按 model signature batch。Jacobian 可以解析实现或由可审计的 torch functional transform 在 compile 阶段生成，但时间步不能构造 Python autograd graph 作为生产 Newton 主路径。

### 4.2 首批器件范围

- `Diode`：Shockley conduction、series R、junction C 的受控简化模型，包含 `pnjlim` 类限制；温度固定为 compile parameter。
- `PiecewiseLinearIV`：电压 knots 严格递增，默认连续线性插值；斜率负值允许但 diagnostics 标记潜在主动/多稳态。
- `PolynomialIV`：有界工作域和显式 extrapolation policy。
- `VoltageDependentCapacitor`：直接声明单值 Q(V)，以 `dQ/dV` 进入 companion/Jacobian；不接受仅给 C(V) 而未定义一致 charge 的歧义模型。

后续候选：`BJT`（Ebers-Moll/Gummel-Poon 受限子集）、`MOSFET`（先 Level-1/平滑 charge-conservative 模型）。每个候选是独立 phase gate，不因 parser 识别 model card 就宣称支持。

### 4.3 求解状态

- `NonlinearMNASystem`：residual、Jacobian stamp plan、scaling、constant linear block。
- `NonlinearSolveState`：当前 iterate、历史 charge、limiting state、last accepted solution。
- `NonlinearSolveStats`：每步 iterations、residual norms、line-search reductions、condition estimate、failure device/node。
- `NonlinearCheckpointState`：accepted solution 和全部 history/limiting，不保存临时 Newton iterates。

## 5. 数值方案与 EM 强耦合

### 5.1 DC operating point

先用 source stepping + gmin stepping 的显式配置求 DC op；每一级 Newton 需要 residual/Jacobian 收敛双判据。gmin 仅是 continuation，最终系统必须回到请求值；最终未收敛则报错，不能把含人工 gmin 的状态当结果。

### 5.2 Transient Newton

对每个 FDTD step，以前一步 accepted state 为初值，C/Q 使用 `04` 选定的 trapezoidal 或 BE companion。组装包含 EM port Norton relation 的完整 residual，Newton 解 `J delta=-r`，应用 device limiting 和 backtracking，直到：

- scaled residual 同时满足 absolute/relative tolerance；
- update norm 满足 voltage/current tolerance；
- 所有器件值 finite 且 charge/current 约束满足。

只有收敛后才 commit field/circuit history。失败时保留 last accepted checkpoint 并报告 time、step、node、device、residual 与迭代轨迹摘要。

### 5.3 时间步与 breakpoint

FDTD `dt` 仍由 CFL 决定。器件切换、PWL kink 和 source discontinuity 注册 breakpoint；若 breakpoint 不落在 dt 网格，第一版对 source/器件在固定 dt 内做一致插值/积分，不单独推进电路多个 substeps，因为这会破坏 EM 同步。确需 subcycling 时作为后续设计，不能隐藏在实现中。

### 5.4 能量与谐波

每步记录 EM port work、器件耗散和储能变化。非线性会产生带外谐波，用户请求的 DFT frequencies 不限制时域物理；benchmark 必须提高网格/time sampling 直到主要谐波低于 Nyquist，并对未观测带外能量给 diagnostics。

## 6. GPU-first、PyTorch 与可微设计

- linear block、nonlinear batched evaluate、Jacobian assembly、small solve、line search 和 convergence reduction 全部在 GPU；只在仿真结束或失败时把摘要送 CPU。
- 为避免每轮 kernel launch 过多，同类型 device fuse/batch，固定 `max_iterations` 的控制可由 CUDA Graph/设备端 active mask 实现；已收敛 batch 不再改变状态。
- device parameters 可为 `torch.Tensor/nn.Parameter`；参数域检查不能 detach 后丢失 graph。
- backward 对每个 accepted implicit solve 使用 implicit-function/transpose-Jacobian solve，并通过 checkpoint replay 重建 forward state；不反向穿过 Newton iteration 路径，也不依赖迭代次数可微。
- 若 forward 接近奇异 Jacobian、多稳态跳变或未收敛，梯度标为无效并报错；不返回有限但错误的梯度。
- PiecewiseLinear kink 处梯度按明确 subgradient policy，默认在 knot 精确命中时报不可微 diagnostics。

## 7. Multi-GPU ownership 与 reduction contract

- 完全沿用 `04`：Circuit owner 持唯一 nonlinear state/Jacobian/Newton stats，port shards 只提供 Norton contributions 和接收 converged currents。
- 每个 Newton iteration 可能需要更新 port voltage-current coupling；EM history 在本 time step 固定，初次 reduction 后缓存于 owner，Newton 内不反复 collective。只有最终 converged currents 每 step scatter 一次。
- Circuit 不跨 rank 拆分，避免 distributed Newton small solve。多个独立 circuits 可按绑定端口 locality 分配不同 owners，从而并行。
- task-level multi-GPU sweep 每 run 独立 DC op/history；禁止共享 mutable limiting state。
- backward 的 circuit adjoint 在 owner 执行，端口 seeds 依 `04` 反向 scatter/reduce。
- 在 distributed adjoint 完成前，非线性参数 trainable + spatial multi-GPU 明确拒绝；forward multi-GPU parity 必须先完成。

## 8. Phases、交付物与 exit gates

### Phase 0：非线性 contract 与 Newton 核心（E0, experimental）

交付：residual/Jacobian protocol、scaling、GPU Newton、line search、DC continuation、diagnostics；用解析标量/小系统 fixture。  
依赖：`04` LinearMNASystem。  
Exit gate：一/多变量非线性方程 corpus 收敛到 double reference；residual/update gate 无假收敛；不收敛 fixture 在限定迭代内给出确定错误；GPU profiler 无每 iteration CPU sync。

### Phase 1：Diode 与静态 I-V 器件（E1, experimental）

交付：Diode/PWL/Polynomial、DC op、transient、SPICE `D` 受限解析和 model diagnostics。  
依赖：Phase 0。  
Exit gate：diode I-V 在工作域相对解析误差 `< 1e-5`；半波/全波整流器波形对固定 Xyce reference normalized RMS `< 1%`；所有 converged steps KCL residual `< tolerance`；不同初值得到预期稳态或报告多稳态。

### Phase 2：非线性电荷、varactor 与 FDTD 强耦合（E2）

交付：Q(V) contract、VoltageDependentCapacitor、junction C、包含 EM Norton relation 的同一步 Newton、energy audit。  
依赖：Phase 1、`04` EM coupling。  
Exit gate：非线性 capacitor charge conservation 相对误差 `< 1e-4`；varactor 谐波频率/幅值对 oversampled reference `< 2%`；RF rectenna 场+电路能量不平衡 `< 3%`；无一拍 phase artifact。

### Phase 3：生产收敛策略与 SPICE workflow（E2）

交付：source/gmin stepping、device limiting、breakpoints、checkpoint resume、batching、支持表和失败报告。  
依赖：Phase 2。  
Exit gate：收敛 corpus（强驱动 diode、限幅、反向恢复简化场景）达到 `>= 99%` accepted steps 且剩余明确失败；checkpoint resume 数值 `rtol <= 1e-6`；失败报告定位到器件/节点。

### Phase 4：可微与 multi-GPU（E3）

交付：implicit backward、参数 tensor、owner/scatter、性能优化。  
依赖：distributed forward contract、Phase 3。  
Exit gate：Is/ideality/PWL slope/Q coefficients 对损失的梯度相对有限差分 `< 2%`（远离 kink/bifurcation）；单/多 GPU transient `rtol <= 2e-5`；32-device circuit 的代表 FDTD step 开销相对线性 MNA `< 2x` 且有 profiler breakdown。

### Phase 5（独立 go/no-go）：BJT/MOSFET 基础模型（E2）

交付候选：BJT 受限模型、charge-conservative Level-1 MOSFET、model-card 参数映射与单独支持矩阵。  
进入条件：Phase 0-4 生产 gate 全通过，且有明确目标用例。  
Exit gate：DC curves、transient switching、charge conservation 分别对 reference `< 2%`；所有支持参数有范围、单位和导数测试。若 gate 未过，不影响 Phase 0-4 作为完成的 diode/behavioral 非线性能力发布，但不得宣传 transistor support。

## 9. 验收策略与 benchmark

### 9.1 单元/数值

- 每个器件 I/Q 及一阶导数对 finite difference/complex step（适用时）。
- Jacobian stamp 对 dense autograd reference；scaling、limiting、line search 和 continuation 状态机。
- KCL/KVL、charge conservation、passive dissipation、active PWL 的功率符号。
- NaN/overflow、极端指数、参数越界、奇异 Jacobian 和最大迭代失败。

### 9.2 端到端 benchmark

1. `nonlinear/diode_iv_dc`：DC curve 与导数。
2. `nonlinear/half_wave_rectifier`：时域、纹波和平均功率。
3. `nonlinear/bridge_rectifier`：多器件 Newton/breakpoint。
4. `nonlinear/diode_limiter_interconnect`：强反射下限幅。
5. `nonlinear/varactor_harmonic_generation`：Q(V)、二/三次谐波。
6. `nonlinear/rectenna`：天线端口到 DC load 的完整 EM-circuit 闭环。
7. `nonlinear/multi_gpu_split_port`：owner contract 和单 GPU parity。

离线 reference 固定 Xyce/ngspice 版本、options、timestep；测试仓库保存数值 golden，不让 CI 依赖外部 solver。

### 9.3 梯度与性能

梯度至少覆盖整流 DC power 对 Is/Cj/load，以及 harmonic amplitude 对 varactor Q coefficients。每项使用多个 finite-difference steps 并避开不可微工作点。性能报告按 device count、unknown count、平均/P95/max Newton iterations、failed line searches、step time、communication 和 checkpoint bytes 分层。

## 10. 风险与缓解

- **指数 overflow/假收敛**：稳定 `expm1`/限制、双 residual+update gate、finite 检查。
- **多稳态/分岔使梯度不可靠**：记录 operating branch；Jacobian condition/bifurcation diagnostics；不在奇异点返回梯度。
- **Newton 与 FDTD 相互反馈不稳定**：解完整同一步 residual，commit-on-convergence，使用能量 gate；不采用显式一拍耦合。
- **器件模型范围无限扩大**：首版固定 diode/PWL/polynomial/Q(V)；transistor 为独立 go/no-go phase。
- **GPU 小系统利用率低**：器件 batching、固定 stamp、run/circuit batch；用数据决定 custom fused solver。
- **SPICE reference 差异**：文档固定模型方程、温度和默认值，不用同名暗示完整厂商语义。

## 11. 完成定义

核心完成指 Phase 0-4 exit gates 全通过：支持的 nonlinear devices 在 `Scene -> Simulation -> Result` 中真实参与 EM time step；DC/transient/能量/梯度 benchmark 达标；Newton 全程 GPU-resident且失败不静默；checkpoint 与单/多 GPU forward parity 达标；API、模型方程、支持范围和 diagnostics 有文档；`FEATURE_LIST.md` 更新。Phase 5 transistor 支持按独立 gate 发布，未完成时必须明确列为非支持能力。
