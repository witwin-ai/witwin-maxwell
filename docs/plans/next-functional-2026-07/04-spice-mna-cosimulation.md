# SPICE/MNA 瞬态协同开发计划

状态：reopened-for-evidence（2026-07-18 审计；契约层与强耦合运行时已落地，多场景守恒/独立求解器交叉验证欠账）  
声明证据级：E3 production；实测证据级：**E1–E2**  
复核结论：见 `docs/assessments/next-functional-audit-2026-07-18.md` §1.3；完成记录不删除，按无通胀规则在文末追加"证据级实测"栏  
原状态（存档）：Phase 0-4 完成（四项 Phase-4 gate 全部有证；详见文末 2026-07-18 完成记录）  
日期：2026-07-14  
目标证据：E3 production  
依赖：`01-rf-engineering-workflow.md` 的 Lumped/TerminalPort、V/I/功率约定和线性 RLC  
Owner modules：`circuits.py`（拟新增）、`compiler/`、`fdtd/`、`result.py`  
最近架构决策：2026-07-14，采用原生 GPU MNA 与同一步强耦合，不以外部 SPICE 进程作为运行时

## 1. 背景与当前能力

`01` 计划中的 RLC 是局部、固定拓扑的高性能端口/集总负载，足以完成 RF 基础闭环，但不能表达带内部节点、独立/受控源、跨多个 EM 端口的电路网络，也不能导入 SPICE netlist。当前 Maxwell 没有 circuit graph、modified nodal analysis (MNA)、DAE 初始化、companion model、场-电路强耦合或电路结果对象。

本计划建立原生 PyTorch/GPU MNA runtime。它不调用外部进程逐步求解，而是在同一个 FDTD time step 内把电路端口电压、电流与 Yee 更新隐式耦合。`01` 的 RLC 编译器最终应复用同一套线性 stamp/state primitive，避免两套离散公式长期漂移。

## 2. 目标与非目标

### 2.1 目标

1. 提供一等 `Circuit` 场景对象、节点/器件/EM port binding 和受控 SPICE 子集导入。
2. 用 MNA 支持线性 R/L/C、独立电压/电流源、VCVS/VCCS/CCVS/CCCS、理想开关的预定时序版本和互感器件。
3. 提供 DC operating point（线性阶段）、transient companion models、初值和一致 DAE 初始化。
4. FDTD 每一步在 GPU 上强耦合求解电路与场，不引入 CPU fallback 或一拍显式反馈。
5. 保持 `Scene -> Simulation -> Result`、`SceneModule`、PyTorch autograd，并定义 multi-GPU circuit owner/reduction。
6. 为 `05-nonlinear-circuit-devices.md` 提供 residual/Jacobian/Newton 扩展接口，但本计划只验收线性电路。

### 2.2 非目标

- 不追求完整 SPICE 方言兼容；第一版不支持厂商 encrypted model、Verilog-A、噪声、Monte Carlo、温度 sweep、`.control` 脚本。
- 不实现非线性 diode/BJT/MOSFET；由 `05` 承接。
- 不实现 harmonic balance、周期稳态或 RF circuit envelope solver。
- 不依赖 ngspice/Xyce 作为生产 runtime；它们仅可用于离线 reference。
- 不允许电路节点直接引用任意 field sample；EM 连接只经过已校准的 Lumped/TerminalPort。

## 3. 用户功能与 Public API 草案

```python
import witwin.maxwell as mw

circuit = mw.Circuit(name="matching_network")
vin = circuit.node("vin")
vout = circuit.node("vout")
gnd = circuit.ground

circuit.add(mw.Resistor("r1", vin, vout, resistance=50.0))
circuit.add(mw.Capacitor("c1", vout, gnd, capacitance=0.8e-12))
circuit.add(mw.Inductor("l1", vin, gnd, inductance=2.2e-9))
circuit.bind_port("feed", positive=vin, negative=gnd)
circuit.bind_port("antenna", positive=vout, negative=gnd)

scene.add_circuit(circuit)
result = mw.Simulation.fdtd(scene, frequencies=freqs).run()
cdata = result.circuit("matching_network")
```

SPICE 子集导入：

```python
circuit = mw.Circuit.from_spice(
    "match.cir",
    name="match",
    parameters={"Cmatch": 0.8e-12},
)
circuit.bind_port("feed", positive="in", negative="0")
scene.add_circuit(circuit)
```

公共契约：

- `Circuit` 属于 Scene 物理定义，使用 `Scene.add_circuit(...)`；运行只由 `Simulation.fdtd(...)` 进入并返回 `Result`。
- `Scene.compile_circuits(...)` 负责编译 topology/stamps/device state；内部使用 `compile_circuit_graph`、`compile_mna_system` 等 `compile_*` 名称。
- `Result.circuit(name)` 返回 `CircuitData`，包含 time/frequency domain node voltage、branch current、device power/energy 和 convergence diagnostics。
- `Circuit.from_spice` 只解析受支持声明，遇未知器件/directive 默认 hard error，不能静默忽略。
- `01` 的 `Resistor/Capacitor/Inductor` 是共享公共器件类型：既可作为 scene-local lumped element，也可加入 Circuit；不建立兼容 wrapper。

## 4. 数据模型

### 4.1 公共对象

- `CircuitNode(name)`：ground id 固定且唯一；用户 name 与编译 index 分离。
- `CircuitDevice` protocol：name、terminals、parameters、initial condition；具体公共类 R/L/C、独立/受控源、MutualInductor、TimedSwitch。
- `PortBinding`：EM port name 与 circuit positive/negative nodes，方向严格对应端口 V/I 定义。
- `Circuit`：有序 nodes/devices/bindings、parameter table、initial condition policy 和 metadata。
- `MNAConfig`：integration (`trapezoidal` 默认或 `backward_euler`)、pivot tolerance、regularization policy、initialization 和 diagnostics level。
- `CircuitData`：`times`、`node_voltages[T,N]`、`branch_currents[T,B]`（按请求采样）、frequency projections、device powers、energy balance、iteration/factorization stats。

### 4.2 内部对象

- `CircuitGraph`：规范化节点、branch unknown 和 source dependency DAG。
- `CompiledStampPlan`：每个器件写入固定 MNA sparse/dense 小矩阵的位置及系数 tensor。
- `LinearMNASystem`：常量/随时间右端、历史状态、端口 incidence matrix、factorization cache。
- `EMCircuitCoupling`：把 EM port 离散等效关系写入 MNA，并把 branch current scatter 回 field update。
- `CircuitCheckpointState`：capacitor voltage、inductor current、source phase、switch schedule index 和必要 replay metadata。

所有 tensor shape 在 compile 后固定。小电路初版可使用 batched dense `torch.linalg.solve`；超过配置阈值（如 256 unknowns）必须显式拒绝或选择已实现的 GPU sparse backend，不能悄悄转 CPU。

## 5. SPICE 子集与语义

第一生产子集：R、L、C、K、V、I、E/G/F/H 受控源、参数表达式的安全 AST、`.param`、`.include`（显式根目录沙箱）、`.subckt/.ends`、`.ic` 和 transient source PULSE/SIN/PWL。节点 `0` 为 ground，单位 suffix 与 SPICE 常用语义固定测试。

不执行任意表达式或 Python `eval`。subcircuit 在 compile 时扁平化并保留层级名字。重复器件、floating voltage-source loop、current-source cutset、奇异矩阵、无 ground、端口绑定极性错误在 prepare 阶段诊断。

## 6. 数值架构与场-电路耦合

### 6.1 MNA 与 companion models

线性 R/受控源直接 stamp；C/L 采用与全局 `dt` 一致的 trapezoidal companion model，startup 或不连续点可局部切换 backward Euler 抑制 ringing，并在 metadata 记录。独立源在 CUDA 上向量化求值。互感 stamp 验证 `|k| <= 1` 与电感矩阵正半定。

`dt` 由 FDTD CFL 决定，circuit 不允许独立大步长跨越场更新。若电路时间常数远小于 `dt`，隐式 MNA保持稳定但精度 diagnostics 必须警告；不以稳定性宣称准确性。

### 6.2 同步强耦合

每个 EM port 的离散 E update 可写成局部 Norton relation `i_em = g_em v + h(fields/history)`。将所有 port incidence 与 circuit MNA 合并为每步小型 Schur system：

1. 各端口从已更新 H/旧 E 构造 history term；
2. GPU solve 同时得到 circuit node voltages 与 EM port voltages；
3. 得到 branch/port currents；
4. scatter currents 更新端口支撑的 E edges；
5. 更新 C/L history 与 observer。

直接项同一步求解，禁止 `V[n] -> circuit -> I[n+1]` 的显式一拍延迟。无 circuit 的现有 FDTD 路径保持原 kernel 和性能。

### 6.3 DC/initial conditions

在 t=0 先解线性 DC operating point：C 开路、L 短路、时域源取初值；用户 `.ic` 可设 constraint 或 initial guess，语义显式。与 EM field 不一致的初始储能必须在 first-step energy report 中显示；无法得到一致初值时阻止运行。

## 7. GPU-first、PyTorch 与梯度

- topology/parser 在 CPU 控制面；stamp values、history、source vectors、factor/solve 和端口 scatter 均驻留 CUDA。
- 固定线性矩阵在每个 dt/config/model version 只 factor 一次；若参数 trainable，使用 differentiable solve，不复用失效 factor。
- R/L/C/source amplitude 可为 CUDA tensor/`nn.Parameter`；编译器保持 dtype/device/grad，不调用无条件 `float()`。
- FDTD adjoint checkpoint 包含 circuit history；backward 对每个离散 MNA solve 使用转置 solve，并与 field adjoint 在端口处交换 seed。
- netlist file 的数值常量默认不可微；用户可通过 `parameters={name: tensor}` 覆盖为 trainable tensor。
- CUDA Graph 仅在 source schedule 和 matrix shape 固定时启用；动态 Python callbacks 不支持。

## 8. Multi-GPU ownership 与 reduction contract

- 每个 Circuit 有唯一 owner rank，按绑定 port reference point 的最小 global index 决定；owner 持唯一 MNA state、factorization 和 source state。
- 每个 EM port 按 `01` 先 reduction 出局部 Norton history/voltage contribution到 circuit owner；owner 每步执行一次 MNA solve。
- 求得的 port currents 只 scatter 到相应 port owner，再由端口 compiler 写入 shard-local edges。通信量为 `O(number_of_bound_ports)` 标量，不复制 circuit matrix 或全场。
- Circuit 内部节点不跨 rank 分割；这是避免每步分布式小矩阵 solve 的明确设计。超大电路不靠空间 FDTD shard 并行，后续由 circuit batching/sparse GPU 解决。
- circuit 与全部绑定 port 同 shard 时必须走零 collective fast path。
- task-level 多 GPU 的每个 run 有独立 circuit state；参数 tensor 的 gradient 在 run-level 聚合后用 torch reduction。
- spatial multi-GPU backward 在 distributed adjoint 完成前精确拒绝；不得只返回 EM 或 circuit 的部分梯度。

## 9. Phases、交付物与 exit gates

### Phase 0：Circuit graph、规范与 SPICE parser（E0, experimental）

交付：公共对象、safe parser、subckt flatten、拓扑诊断、共享 RLC 类型决策。  
依赖：`01` 端口/RLC API freeze。  
Exit gate：受支持 corpus 解析/序列化一致；所有 unsupported directive hard error；节点/branch ordering deterministic；恶意表达式不能执行代码。

### Phase 1：独立线性 MNA runtime（E1, experimental）

交付：RLC/源/受控源/互感 stamp、DC op、trapezoidal/BE transient、CircuitData。  
依赖：Phase 0。  
Exit gate：RC/RL/RLC step、二阶网络、变压器与受控源相对解析/Xyce reference `L_inf < 1e-4`（双精度离线 gate）；离散能量误差随 dt 二阶收敛；奇异拓扑诊断准确。

### Phase 2：单端口 FDTD 强耦合（E2）

交付：EMCircuitCoupling、GPU Schur solve、checkpoint、端口功率与 energy audit。  
依赖：Phase 1、`01` Lumped/TerminalPort。  
Exit gate：同一个 RLC 作为 native load 与 Circuit 的 S11 差 `< 1%`/`< 2 deg`；场+电路能量不平衡 `< 2%`；长时间无被动增能；profiler 无逐步 CPU transfer。

### Phase 3：多端口与 SPICE workflow（E2）

交付：多 port binding、受控源跨端口、PULSE/SIN/PWL、保存加载和端到端文档。  
依赖：Phase 2。  
Exit gate：二端口 matching/filter network 与独立全电路 reference S 参数差 `< 0.02`；FDTD 稳态与电路频率响应差 `< 2%`；checkpoint resume 后结果 `rtol <= 1e-6`。

### Phase 4：PyTorch 梯度、multi-GPU 与性能（E3）

交付：linear MNA adjoint、tensor parameters、owner/reduction、batched small solves/CUDA Graph 支持。  
依赖：multi-GPU forward contract、Phase 3。  
Exit gate：R/L/C/源参数梯度对有限差分相对误差 `< 1%`；单/多 GPU `rtol <= 2e-5`；代表 32-unknown circuit 的 FDTD step 开销 `< 10%`；无 circuit 的单 GPU回退 `< 1%`。

## 10. 验收策略与 benchmark

### 10.1 单元/数值

- 每类 stamp 与手写矩阵逐项比较；ground elimination、branch ordering、参数单位和 source waveform。
- DC op、初值、switch breakpoint、trapezoidal history、energy/power sign。
- dense solve 的 condition diagnostics、pivot threshold 和 dtype。
- save/load、device migration、`SceneModule.to_scene()` 保持参数 graph。

### 10.2 端到端 benchmark

1. `circuit/rc_lowpass`：解析 transient/frequency response。
2. `circuit/series_parallel_resonator`：与 `01` native RLC parity。
3. `circuit/transformer_coupled_load`：K stamp 和功率。
4. `circuit/transmission_line_matching_network`：两个 EM ports + 多节点电路。
5. `circuit/pulse_driven_interconnect`：PULSE、反射和 checkpoint resume。
6. `circuit/multi_gpu_split_bindings`：owner/scatter parity。

reference 由解析公式和固定版本 Xyce/ngspice 离线生成，记录 netlist、版本、tolerances；生产测试不依赖外部工具安装。

### 10.3 梯度/性能

梯度验证至少覆盖 RC cutoff、RLC resonance、两端口 insertion loss 和端口附近 EM 材料参数。性能记录 unknown count、factor/solve time、每 step communication、state/checkpoint bytes；分别报告 8/32/128 unknowns，不用单一小电路掩盖复杂度。

## 11. 风险与缓解

- **代数环/奇异 MNA**：prepare 阶段结构秩诊断，运行时 condition 监测；不默认加泄漏电导掩盖拓扑错误。
- **trapezoidal ringing**：在不连续点显式 BE startup/local damping 并记录，不全局偷偷换算法。
- **EM/circuit sign 错误**：统一 `01` 的 V/I orientation，以匹配负载、能量守恒和方向反转三重 gate。
- **小矩阵 solve 阻塞 GPU**：固定矩阵预分解、batch 多 circuit/run、避免 CPU launch loop；用 profiler 决定自定义 kernel 门槛。
- **SPICE scope 膨胀**：维护公开支持表；未知 directive hard error；完整方言不是完成条件。
- **与 01 RLC 重复**：共享 device classes/stamp primitives，native local fast path 是同一模型的优化 lowering。

## 12. 完成定义

Phase 0-4 exit gates 全通过；用户能把受支持 `.cir` 或 Python Circuit 绑定到一个/多个 EM ports 并通过 `Scene -> Simulation -> Result` 得到电路/场结果；线性强耦合真实参与每个 FDTD step；GPU profiler 无逐步 CPU fallback；梯度与 multi-GPU contract 完整；unsupported SPICE 输入不被静默接受；benchmark 进入 CI；新增能力更新 `FEATURE_LIST.md`。

## 13. 修订记录（append-only，不重写历史）

上文 §1-§12 正文保留原样以存档 proposed 阶段决策；本节仅追加完成状态与证据引用，如与正文冲突以本节为准。

### 2026-07-18 完成记录（Phase 0-4 完成，master `3f13ff2`）

Phase 0-3 已完成并落地。Phase 4 四项 exit gate（§9 Phase 4）全部有证，验收记录为 `docs/assessments/spice-mna-phase-4-acceptance.md`（Status: accepted，4 of 4 exit gates evidenced）。逐项结算：

- **gate (a) 参数梯度 vs 有限差分 `< 1%` — PASS。** `tests/gradients/test_fdtd_circuit_adjoint.py`（25 passed），全部端到端用例经三步中心差分 `max_relative_error=0.01`，覆盖 R/L/C/源幅度、RC cutoff、RLC 近共振、二端口插损、绑定端口材料参数、直接张量与 `SceneModule` 参数。
- **gate (b) 单/多 GPU parity `rtol <= 2e-5` — PASS（bitwise）。** `tests/fdtd/multi_gpu/test_circuit_owner.py`（7 passed），六个场量、端口 V/I 与电路 node/branch 数据在两 shard 场景上 bitwise 相同（偏差 0）；scalar-transfer contract 为 `O(bound_ports)`、每步 8 字节。
- **gate (c) 代表 32-unknown 电路 step 开销 `< 10%` — PASS（matched 定义）。** artifact `docs/assessments/spice-mna-phase-4-circuit-performance.json`：32-unknown 相对匹配的原生端接基线 **-64.507%**；8/32/128 未知量成本平坦（单次 factorization），MNA solve 非瓶颈。验收文档诚实记录 caveat：该负值来自匹配基线（原生 SeriesRLC 端接）自身偏慢，非“电路协同接近免费”。
- **gate (d) 无电路单 GPU 回退 `< 1%` — PASS（等价证据为主）。** 以主机指令流等价为主证据：`docs/assessments/spice-mna-phase-4-no-feature-op-stream.json`（512 步 op 表逐项相同，3282 aten calls / 50 keys，empty diff，`equivalent=true`）；计时 leg 用 A/A 校准归档而非挑绿数字：`-abba.json`（clock-floored B-run +1.227%）与 `-aa-calibration.json`（同一 commit 两 checkout 相差 -0.523%，即单次分辨率粗于 1%）。基线为不可变 Phase 3 树 `0a69fc8`，按 Git archive 内容哈希校验。

#### 端口接口的梯形统一（trapezoidal port-interface unification）

端口-场耦合在原生 lumped、MNA 电路与嵌入网络三条路径上统一为同一梯形（trapezoidal）半步电压接口。跨模型精确性证据：`tests/rf/circuits/test_fdtd_circuit_coupling.py`（native.Ez vs coupled.Ez，`rtol=2.0e-6`）、`tests/rf/circuits/test_phase3_multiport.py`（`rtol=2.0e-6`）。

#### 电路热路径工作（circuit hot-path）

Phase 4 落地了电路热路径的 op-count 契约与图路径：CUDA Graph 在固定 source schedule / matrix shape 下启用（32-unknown 实测 eager 1.276 → graph 0.332 ms/step，3.849x），固定线性矩阵每 dt/config 只 factor 一次。相关 op 计数与 graph 路径证据见 `spice-mna-phase-4-circuit-performance.json` 及 `docs/assessments/port-hot-path-op-inventory-2026-07-17.json` / `port-hot-path-timing-2026-07-17.json`。

#### 性能契约再资格化（re-qualification）

原契约在另一 host（单 RTX 5080）冻结，已按 §11 变更规则在本 host（2x RTX A6000）重测并记录旧值/新值/技术理由，`< 10%` 与 `< 1%` 门限本身不变（详见验收文档 “Performance contract re-qualification” 节）。

### 2026-07-18 证据级实测（审计回退，不删除上文完成记录）

本栏依 `docs/assessments/next-functional-audit-2026-07-18.md` §1.3 与 §4 的无通胀规则追加。上文"完成记录"逐条保留存档；本栏只登记**实测证据级与欠账**，如与上文的完成声明冲突，以本栏对证据级的判定为准。

- **实测证据级：E1–E2**（非声明的 E3）。原生 GPU MNA、同一步强耦合、companion model、pivoted-LU、CUDA Graph replay 与单卡伴随确实落地并进 `FEATURE_LIST`；参数梯度 vs 三步中心差分（gate a）与单/多 GPU bitwise parity（gate b）为可信的 E2 级证据。上封顶在 E2、未达 E3 的原因是缺**多场景守恒/能量残差**与**独立电路求解器（离线参考）交叉验证**。
- **端到端 EM+电路瞬态强耦合无外部对照**：Tidy3D 不覆盖 FDTD+SPICE 强耦合（审计 §3 参考策略表 04 行），当前只有自洽性（跨路径梯形接口 `rtol=2e-6`）与解析 RC/RLC 门；端到端强耦合门必须标 `reference: future-xfdtd`。
- **继承 01 端口功率约定风险**：Lumped/TerminalPort V/I/功率约定继承自 `01`，上游未经 wave 级验证。
- **gate (c) 性能为 matched 定义**：`-64.507%` 来自匹配基线（原生 SeriesRLC 端接）自身偏慢，非"电路协同接近免费"；上文已诚实记录 caveat，此处确认其不构成裸 FDTD 相对的性能证据。
- **提升到 E2/E3 所需证据（收敛路线，见审计 S3.2）**：
  1. 多场景守恒/能量残差 + **独立电路求解器（离线）** 交叉验证（提 E2）；
  2. 端到端 EM+电路强耦合门标 `reference: future-xfdtd` 并以解析/守恒占位，不得自证或跳过；
  3. 组合矩阵、命名硬件性能边界、分布式/梯度声明与公开 benchmark 进入 RESULTS（提 E3，README §7 定义）。
- 进入门：本计划 S3.2 收敛工作阻塞于 S1（01 端口 wave 级验证）先行通过。
