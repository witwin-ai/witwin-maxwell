# 数值门分类规范（Gate Classification Spec）

> 状态：normative（S0.3 交付，2026-07-18）
> 来源：`docs/assessments/next-functional-audit-2026-07-18.md` §2（系统性缺陷模式）与 §4（勾销的硬性定义）
> 适用范围：本仓所有计划 phase 的 exit gate、benchmark 判定门与验收文档中声明的"数值门"

## 0. 为什么需要这份规范

2026-07-18 审计发现全路线反复出现"证据级通胀"：一个测试断言的量，恰恰是被测代码定义式计算出来的量；一个"物理不变量"其实由对称 fixture 自动保证；一个"端到端"其实是把解析输入喂进后处理。这些门形式上 PASS，但**不提供任何独立物理证据**。

因此本规范强制：**每个数值门必须自标一个类别**，且**前四类不得作为某个 exit gate 的唯一证据**。类别标注写在测试 docstring / 验收文档的门表 / benchmark 场景说明里，复核者据此判断证据是否成立。

## 1. 五个类别

| 类别 | 一句话定义 | 能否单独作为 exit gate 证据 |
|---|---|---|
| `analytic-identity` | 断言一个在限定工况下成立的闭式恒等式，求解器未在 wave 级驱动它 | ❌ 否 |
| `tautology` | 断言的量正是被测代码定义式计算的量（自证 / 除以自身） | ❌ 否 |
| `symmetric` | 结果由 fixture 的对称性自动保证，换非对称 fixture 即失去判别力 | ❌ 否 |
| `postprocess-only` | 把解析/构造输入喂进后处理，绕过或 monkeypatch 掉求解器 | ❌ 否 |
| `wave-level` | 求解器在 wave 级真实运行，输出与**独立参考**对照，附收敛证据 | ✅ 是（满足 §3 清单时） |

规则（audit §2、§4）：

1. **前四类是必要的辅助门，但永远不是充分门。** 它们可以捕捉回归、锁定接口约定、验证代数正确性，但不能证明"物理可用"。
2. **一个声明 E2 及以上的 exit gate 必须至少有一个 `wave-level` 门作为头条证据。** E0/E1 可以只有前四类，但必须显式标 experimental，且不得声称物理可用。
3. **禁止用类别混淆冒充等级。** 例如把 `analytic-identity` 写成"FDTD 数值门"、把 `symmetric` 写成"物理互易验证"，即构成证据通胀，复核直接判 FAIL。
4. **多个非 wave-level 门叠加不升级为 wave-level。** 三个 tautology 门仍是 tautology。

## 2. 各类别的工作示例（引自审计实测）

以下每个反例均取自 `next-functional-audit-2026-07-18.md` 对计划 01 的逐 gate 复核，file:line 为审计锚定的实测出处。

### 2.1 `analytic-identity`（反例）

- **门**："matched load `|S11| < −30 dB`"。
- **实测**：`tests/rf/lumped/test_fdtd_lumped_runtime.py:208-235` 施加一次隐式更新后检查 `V = Z0·I` 的构造对（`Γ = 0` at `atol 1e-12`），**无波、无网格、无频窗**。
- **同类**："open/short 幅值 `<0.02` / 相位 `<3°`"为极限恒等式（`Γ = ±1` within `1e-12`）；"串/并联 RLC 谐振 `<2%`"由 `tests/rf/lumped/test_rlc_discretization.py:83-97` 扫**梯形阻抗公式**对解析谐振，**FDTD 未运行**。
- **为何不充分**：限定工况的闭式恒等式在 `atol 1e-12` 恒成立，与 FDTD 是否正确无关。

### 2.2 `tautology`（反例）

- **门**：Phase 3 "TEM 功率归一化 `<1%`"。
- **实测**：归一化量除以被测函数本身，差恒为零（审计 §1.1 Phase 3）。
- **同类**："LumpedPort vs WavePort S 差 `<0.03`"把两者推入同一 `V/I` 相量，差**构造上恒为零**；Phase 5 realized gain 恒等式同理。
- **为何不充分**：断言 `x == x`，无外部信息量。

### 2.3 `symmetric`（反例）

- **门**："coax/microstrip 互易 `<0.02`"。
- **实测**：fixture 是 `8³` 真空盒里两个**镜像对称**的 lumped port（`tests/rf/network/test_network_sweep.py:39-52,326-369`），对称性自动保证 `S12 = S21`；`benchmark/scenes/` 无任何 coax/microstrip。
- **同类**："功率不平衡 `<2%`"由 `tests/rf/network/test_network_algebra.py:11-27` 对**手写酉矩阵** assert，零求解内容。
- **为何不充分**：换成几何/材料非对称的二端口即失去判别力；对称 fixture 不检验物理互易。

### 2.4 `postprocess-only`（反例）

- **门**：Phase 4 天线 gain / realized gain / 轴比 / 方向图。
- **实测**：喂教科书方向图 / 构造 CP 场（`tests/rf/antenna/test_antenna_data.py`），**无任何 FDTD 天线**；`Result.antenna` 集成测试 **monkeypatch 掉远场计算**（审计 §1.1 Phase 4）。
- **为何不充分**：后处理管道正确 ≠ 端到端物理正确；求解器被绕过。

### 2.5 `wave-level`（正例）

- **门**：Phase 3 "coax TEM `Z0 < 2%`"、"矩形波导 TE10 截止频率 `< 2%`"。
- **实测**：真实电磁 / 模式求解，输出与**解析传输线 / 解析截止频率**对照（审计 §1.1 Phase 3 判定 **成立**）。
- **另一正例**：Phase 5 解析 VJP vs CUDA autograd 到 `2e-13`、逐位 checkpoint replay 成立（伴随机制的 wave-level 证据）。
- **为何充分（在补齐 §3 后）**：求解器在 wave 级真实运行，参考独立于被测实现，且可加收敛报告与 benchmark 常驻。

> 注意：正例 Phase 5 的**机制**是 wave-level，但其**有限差分门**曾靠 min-of-three 通过（port series C 仅 1/3 通过：9.92%/2.46%/0.63%），审计要求硬化为收敛阶检查。即"wave-level 类别"是必要条件，判据统计法仍须另行满足 §3。

## 3. 未来门作者检查清单（audit §4 硬性定义）

一个门要作为 exit gate 的头条证据（对应计划 phase 标 `completed` 的前提），作者必须逐项勾选：

- [ ] **类别自标**：门在 docstring / 门表里显式写 `analytic-identity | tautology | symmetric | postprocess-only | wave-level` 之一。头条门必须是 `wave-level`。
- [ ] **falsification check（可证伪性）**：写明"什么输入会让这个门 FAIL"，并给出一个确实会 FAIL 的对照（如注入已知误差、换非对称 fixture、扰动参数）。若无法构造 FAIL 分支，该门是自证，判 tautology。
- [ ] **independent reference（独立参考）**：至少一类独立于被测实现的参考——解析解 / 独立求解器 / 标准数据 / 守恒律。外部求解器对照遵循审计 §3 "S·参考求解器策略"：外部参考后端覆盖的能力必须用它对照，未覆盖的标 `reference: future-xfdtd` 并以解析/独立求解器占位（不得因此跳过或自证）。
- [ ] **convergence report（收敛报告）**：涉及 PDE / 几何离散的能力，附网格 / 时间步 / 运行长度三档收敛，附守恒 / 无源性检查。
- [ ] **RESULTS presence（基准常驻）**：相关 benchmark 场景**存在且进入** `benchmark/RESULTS.md`，当前值达标或差距如实标注（`WITWIN_BENCHMARK_NO_CLOUD=1` 计分）。
- [ ] **statistics（判据统计法，性能门额外）**：性能门用方差感知判据（如 95% CI 上界 < 目标 + 预热 + 多重复中位数 + 方差报告），不用单点 min / 中位数。
- [ ] **独立复核**：由计划作者以外的复核确认后，方可从 `reopened` / `active` 转 `completed`。

只有以上全部满足，phase 才能标 `completed`；否则如实记录实测证据级（E0–E3，见 `docs/plans/next-functional-2026-07/README.md` §7）与差距，不得以"API 存在"充数。

## 5. 性能门标签族（`perf` label family，非数值门）

第 1 节的五个类别针对**数值/物理门**。运行成本门（op-count 调度计数、方差感知
时间判据）不断言任何物理量，落在那五类之外，因此单独给出一个 `perf` 标签族。此
为 additive 扩展，不改动第 1–4 节对数值门的约束。

| 标签 | 一句话定义 | 断言对象 | 能否作为物理 exit gate 头条证据 |
|---|---|---|---|
| `perf-opcount` | 确定性主机/设备 dispatch 计数（kernel launch、alloc、DtoD、host<->device 拷贝、scalar sync、device mem），不断言墙钟时间 | 调度形状（schedule shape） | ❌ 否（非物理门） |
| `perf-statistical` | 方差感知的时间判据：预热丢弃 + 多重复配对比 + 95% CI 上界 < 目标 + 方差报告，不用单点 min/中位数 | 墙钟性能回归 | ❌ 否（非物理门） |

规则：

1. **`perf` 门永远不是物理可用性证据。** 它锁定成本/回归，不替代第 1 节的
   `wave-level` 头条门；一个 phase 的物理等级仍由数值门决定。
2. **`perf-statistical` 必须满足第 3 节"判据统计法"条目**（95% CI 上界、预热、
   多重复中位数、方差报告），不得退回单点 min/中位数。
3. **`perf-opcount` 必须自带 falsification 记录**：写明什么调度回归会使其变红，
   并给出一个确曾变红的对照（如还原某条 fast-path）。
4. **类别自标写在测试 docstring / 门表**，与第 1 节数值门一致，复核者据此判断。

现有自标示例：`tests/rf/performance/test_port_hot_path_op_count.py` 与
`tests/rf/lumped/test_fdtd_port_end_to_end.py` 的 op-count 天花板门标
`perf-opcount`；`tests/support/perf_variance_gate.py` 与
`tests/rf/performance/test_perf_variance_gate.py` 标 `perf-statistical`。

## 4. 与验收文档的衔接

- 每份 `docs/assessments/*-acceptance.md` 的门表新增一列 `class`，逐门标类别。
- 头条门非 `wave-level` 的 phase 不得声明 E2 及以上；其验收文档 Status 应为 `reopened-for-evidence` 或明确的 experimental。
- 2026-07-18 审计已据本规范对 01/03/04/06/07 回退证据级；后续新计划从第一个 phase 起即按本规范标注，避免事后回退。
