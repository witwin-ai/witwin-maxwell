# next-functional-2026-07 已执行计划审计与严格执行步骤

> 日期：2026-07-18
> 审计范围：`docs/plans/next-functional-2026-07/` 中所有已标记 `completed` / `Phase X 完成` 的计划
> 判定基准：本路线自定义的 E0–E3 证据等级（`README.md` §7）
> 关联文档：`docs/assessments/plan-01-rf-workflow-audit-2026-07-16.md`、`docs/assessments/engineering-gap-analysis-2026-07-12.md`、`docs/assessments/functional-gap-analysis-2026-07-12.md`
> 数据来源：worktree `master`、`FEATURE_LIST.md`、`benchmark/RESULTS.md`（2026-07-18 重跑）、各计划完成记录、`tests/` 实测

## 0. 一句话结论

已执行的计划把 **公共 API、`compile_*` 编译层、结构化 Result、fail-closed guard、参考 oracle** 成体系铺完了（"框架完整"），但**物理内核与 wave-level 数值验证是被系统性推迟的那一半**。多数标记"完成"的计划真实证据等级停在 **E0–E1**，而非其声明的 **E2/E3**。这就是本路线当前的"弱完整完成"状态。

因此：**在把已落地契约从 E0/E1 拉到 E2 之前，不得启动任何 Wave C/D 的新功能物理实现**。

## 1. 逐计划真实状态与欠账

状态标记：✅ 达到声明证据级；🟡 部分达到/网格或场景受限；⚠️ 声明与实测证据不符（overstated）；❌ 仅契约或未消费。

| 计划 | 声明状态 | 实测证据级 | 判定 |
|---|---|---|---|
| 01 RF 工程闭环 | completed / E3 | **E0–E1** | ⚠️ 严重高估 |
| 02 多 GPU 执行 | ensemble 正确性 + joint NCCL 前向 | 前向 E2 | 🟡 前向可信，伴随/计时/组合缺 |
| 03 Touchstone 嵌入 | Phase 0–4 完成 | E1（部分 gate 网格限定 PASS） | 🟡 |
| 04 SPICE/MNA 协同 | Phase 0–4 完成 | E1–E2 | 🟡 |
| 05 非线性器件 | Phase 0 | E0 | ❌ FDTD 耦合 fail-closed |
| 06 阵列/Active-S/MIMO | Phase 0–3 + 权重梯度 | E1–E2 | 🟡 场景梯度 fail-closed；96³ 资格化 deferred |
| 07 Thin-wire | Phase 0–3 完成 / Phase 4 部分 | PEC E2；有损 E0 | 🟡 有损电流递推 fail-closed |
| 08 Ferrite | Phase 0 | E0 | ❌ FDTD 运行时 fail-closed |
| 09 表面阻抗/粗糙度 | Phase 0 | E0 | ❌ adapter fail-closed |
| 10–13 SAR/Bioheat/静电/ESD | proposed | — | 仅设计 |

### 1.1 计划 01 — RF 工程闭环（⚠️ 最大欠账，全路线 P0）

`plan-01-rf-workflow-audit-2026-07-16.md` 已逐 gate 复核，本节汇总其结论并锚定证据：

**根本问题：声明 E3，实测头条数值门多为代数恒等式或后处理契约，FDTD 从未在 wave 级驱动这些门。**

- **Phase 1 头条门是单步代数恒等式**：
  - "matched load |S11| < −30 dB" — `tests/rf/lumped/test_fdtd_lumped_runtime.py:208-235` 施加一次隐式更新后检查 V=Z0·I 构造对（Γ=0 at atol 1e-12），无波、无网格、无窗口。
  - "open/short 幅值 <0.02 / 相位 <3°" — 同类极限恒等式（Γ=±1 within 1e-12）。
  - "串/并联 RLC 谐振 <2%" — `test_rlc_discretization.py:83-97` 扫**梯形阻抗公式**对解析谐振，**FDTD 未运行**。
- **Phase 2 互易/功率门无物理内容**：
  - "coax/microstrip 互易 <0.02" — 实际 fixture 是 8³ 真空盒里两个镜像对称 lumped port（`test_network_sweep.py:39-52,326-369`），对称性自动保证 S12=S21；`benchmark/scenes/` 无任何 coax/microstrip。
  - "功率不平衡 <2%" — `test_network_algebra.py:11-27` 对手写酉矩阵 assert，零求解内容。
- **Phase 3 混合**：coax TEM Z0<2%、TE10 截止<2% **成立**（真实电磁/模式求解）；但 "TEM 功率归一化<1%" 是除以被测函数本身的**同义反复**；"LumpedPort vs WavePort S 差<0.03" 推入同一 V/I 相量，差恒为零。
- **Phase 4 全部后处理契约**：喂教科书方向图/构造 CP 场（`test_antenna_data.py`），无任何 FDTD 天线；`Result.antenna` 集成测试 monkeypatch 掉远场计算。
- **Phase 5 伴随机制过硬但门脆弱**：解析 VJP vs CUDA autograd 到 2e-13、逐位 checkpoint replay 成立；但 FD 门靠 **min-of-three** 通过——port series C 仅 **1/3 通过**（9.92%/2.46%/0.63%）；性能门 1.906% vs 2.0% 落在自身噪声带内。
- **性能欠账**：SeriesRLC 端口每步每端口 **14.4× 开销**（153 aten op / 78 kernelLaunch / 16 DtoD，全在 CUDA graph 外，`fdtd/lumped.py:518-708`、`ports.py:635-659`、`stepping.py:1942,1954`），与 §9.4 目标（<5%/<2%）直接矛盾，零工件。

**计划 01 完整欠账清单（按优先级）**：
1. 端口热路径性能（§9.4 全部目标零工件，且被 14.4× 反证）。
2. 头条门的物理 FDTD 验证：wave 级 matched/open/short S11、FDTD RLC 谐振、真实 coax/microstrip 二端口（§10 九个 RF 基准场景一个都不存在）。
3. FDTD 驱动的天线基准（Phase 4 目前纯后处理）。
4. FD plateau 报告（§9.3）+ 把 min-of-three 硬化为收敛阶检查。
5. 非退化互易 fixture（非对称二端口）+ 物理功率平衡门替换酉矩阵常量 assert。
6. 几何上可区分的 LumpedPort-vs-WavePort 重叠门（当前构造上恒零）。
7. 四类 canonical fresh-process 工作流（§12，无示例脚本）。
8. 性能门统计法与 stale 出处（引用了不存在的 `witwin2` 环境与 Windows 路径）。

### 1.2 计划 02 — 多 GPU 执行（🟡 前向可信，边界大）

- ensemble：N 独立 Simulation 逐端口/扫参正确性 `torch.equal` 复现，失败矩阵/租约/容量预检齐全（`02-ensemble-progress.md`）。**计时/加速比全部 deferred-pending-exclusive-window，无任何数字**。
- joint-solve：one-process-per-GPU NCCL 前向端到端跑通（`02-phase-7-8-blueprint`）。但 `_validate_static_capabilities`（`solver.py:373-472`）**拒绝** Bloch、x-periodic、x-symmetry、MaterialRegion 密度、端口、closed-surface/diffraction/flux-time/非点 time monitor、SIBC、材料 monitor、split x 面上的 Ex、非点/非均匀源——即分布式路径与几乎所有 RF/研究功能互斥。
- **欠账**：分布式伴随反传（deferred）；`DistributedFDTD` 自身缺 trainable guard（防御纵深缺口，blueprint 自述）；单卡/多卡长时程 float32 halo drift 已知；ensemble 与 joint-solve 组合仍被拒。

### 1.3 计划 03/04 — Touchstone 嵌入 / SPICE-MNA（🟡）

- 两者 Phase 0–4 均有完成记录，运行时（有理拟合→状态空间→同步耦合、pivoted-LU、CUDA Graph replay、单卡伴随）确实落地并进 `FEATURE_LIST`。
- **欠账**：03 Phase 4 gate (d) 为**网格限定 PASS**（非通用）；两者缺多场景守恒/无源性/独立求解器交叉验证的 E2 证据；分布式路径大量组合 fail-closed（见 §1.2）；均依赖 01 的端口功率约定，而该约定本身未经 wave 级验证（继承 01 的风险）。

### 1.4 计划 06 — 阵列/Active-S/MIMO（🟡）

- Phase 0–3 + 权重梯度完成，codebook/max-hold/MIMO/ECC 全在 autograd 图内，gradcheck 通过。
- **欠账**：`ArrayBasisData.scene_gradient_vjp(...)` **fail-closed（抛 NotImplementedError）**——场景/材料/几何梯度未贯穿 basis，依赖 02 Phase 7 分布式 result-aggregation；冻结 96³/4096-step 资格化 + Phase 1 门限再资格化 deferred-pending-exclusive-window；全部 EIRP/realized gain 继承 01 的端口功率链风险。

### 1.5 计划 07 — Thin-wire（🟡）

- PEC 直/弯/分支/闭环路径完整（前向+伴随+多 GPU forward，能量一致递推），Phase 0–3 成立。
- **欠账**：有限电导率仅**串联阻抗模型**（解析趋肤 + 无源有理 ADE）落地并过 2% 解析门；**有损电流递推、`ohmic_loss` monitor、电导率伴随、分布式 wire reverse 全部 fail-closed**——即有限导体 wire 编译进 FDTD 会带明确报错拒绝；trainable wire 在多 GPU、分布式 CPML/Mur、wire 与电路/网络混用均 fail-closed。

### 1.6 计划 05/08/09 — 非线性器件 / Ferrite / 表面阻抗（❌ 契约层）

- 均**仅 Phase 0**：类型/契约/参考 oracle/fail-closed 边界落地，**求解器不消费**：
  - 05：Newton 核 + `NonlinearMNASystem` 成立，但 FDTD 耦合、瞬态 companion、伴随、benchmark 是后续 slice；BJT/MOSFET 保留并 fail-closed。
  - 08：`GyromagneticFerrite` + Polder 张量 + 能量无源性证明成立，但 **FDTD 编译器/运行时对 ferrite 结构直接 fail-closed**（否则静默丢弃 gyrotropy）。
  - 09：契约 + 有理模型 + 拟合器 + fail-closed 漏斗 + adapter 导出 fail-closed。

### 1.7 基准可信度（横切所有计划）

`benchmark/RESULTS.md`（2026-07-18 重跑）仍有场景**超出自身 L2<1e-1 目标**：
- `multi_dielectric` L2=0.61、corr=0.83；`custom_current_source` L2=0.80、flux=0.77；`lorentz_resonator` L2=0.27；`modulated_slab` L2=0.46；`high_eps_box` flux=0.79；`sigma_e_drude_slab` L2=0.21。
- **无一个 RF/端口/天线/阵列基准场景**进入 RESULTS（与 01 §10 承诺矛盾）。核心 vacuum/dipole/kerr/debye 达标。

## 2. 系统性缺陷模式（根因）

1. **门自证（tautological gates）**：多处测试断言的量正是被测代码定义式计算的量（01 Phase 3 功率归一化、Phase 5 realized gain 恒等式），无独立参考。
2. **代数恒等式冒充波级验证**：把限定工况的闭式恒等式当作 FDTD 数值门（01 Phase 1 全部）。
3. **对称 fixture 冒充物理不变量**（01 Phase 2 互易）。
4. **后处理喂解析输入冒充端到端**（01 Phase 4 天线、03/04 部分）。
5. **证据级通胀**：Phase exit gate 声明 E2/E3，但只提供 E0/E1 证据；`completed` 标记未经独立复核即写入。
6. **性能门统计不设防**：单点 min/中位数落在噪声带内即判 PASS，无方差/收敛判据。
7. **契约完成 = 计划完成的错觉**：Wave C 的 Phase 0（契约 + fail-closed）被计入进度，掩盖"求解器不消费"这一事实。

## 3. 严格执行步骤（强制顺序，前置门未过不得进入下一步）

以下为**收敛路线**，非新功能路线。每步产出**机器可读工件 + 独立参考 + 落入 CI 分层回归**，方可勾销。

### S·参考求解器策略（reference-solver policy）

所有"独立求解器对照"统一遵循以下优先级（解析解与守恒律始终为第一线，本策略只规定**外部求解器交叉验证**的取用顺序）：

1. **优先 Tidy3D**：凡是 Tidy3D 公开能力覆盖的对照，一律用 Tidy3D，经**已有的 benchmark adapter**（`witwin/maxwell/adapters/`、`benchmark/`，属互操作适配器范畴）生成参考并入库缓存；参考须固定 Tidy3D 版本与生成参数。
2. **Tidy3D 不覆盖的能力 → 标记为"未来用 xFdtd 对照"**：在对应步骤/场景显式写 `reference: future-xfdtd`，当前先以解析解 / 独立专用求解器 / 守恒律占位，不得因缺外部对照而把该门降格为自证或跳过。
3. 不得用 Maxwell→Tidy3D 导出充当 Maxwell 原生已验证；导出成功 ≠ 数值对照通过。

**逐计划对照映射**（用于 S1/S3/S5/S6 选择参考；标 `future-xfdtd` 的项在 Tidy3D 补齐前不阻塞收敛，但必须占位标注）：

| 能力域 | Tidy3D 是否可对照 | 主对照（优先） | 未来次对照 |
|---|---|---|---|
| 01 RF 端口 / N-port S/Z/Y | ✅（Terminal/Lumped/Wave port、N-port S） | 解析传输线 + Tidy3D | xFdtd |
| 01 天线 gain/realized gain/efficiency | ✅（天线指标） | 解析（半波偶极子）+ Tidy3D | xFdtd |
| 03 Touchstone 网络嵌入 | 🟡（lumped/network 有限，需逐项确认） | 独立 S 级联 + Tidy3D（若其 API 支持） | xFdtd |
| 04 SPICE/MNA 强耦合 | ❌（无 FDTD+SPICE 瞬态强耦合） | 独立电路求解器（离线） | **future-xfdtd** |
| 05 非线性电路器件 | ❌ | 独立电路求解器 + 解析 | **future-xfdtd** |
| 06 阵列 / Active-S / MIMO | 🟡（RF 部分，需确认） | 解析阵列因子 + Tidy3D（若支持） | xFdtd（array opt/MIMO） |
| 07 Thin-wire 亚网格 | ❌（无亚网格细线模型） | 解析（趋肤/传输线） | **future-xfdtd** |
| 08 Ferrite / gyrotropic | ❌（无磁化铁氧体） | 解析（gyrotropic 谐振 / Faraday 旋转） | **future-xfdtd** |
| 09 表面阻抗 / 金属粗糙度 | ✅（SIBC + Hammerstad/Huray） | Leontovich 解析 + Tidy3D | xFdtd |
| 10 SAR | ❌（无 SAR 标准工作流） | 解析 | **future-xfdtd**（SAR + IEC/IEEE） |
| 11 Bioheat | 🟡（有 Heat 求解器，SAR→热链需确认） | Tidy3D Heat（若适用）+ 解析 | xFdtd bioheat |
| 12 静电 / 电容 | 🟡（有 Charge/conduction DC） | Tidy3D（若适用）+ 解析电容 | xFdtd 静电 |
| 13 ESD / 介电击穿 | ❌ | 解析 / 独立 | **future-xfdtd** |

其中 🟡 项在动工前必须先做一次"Tidy3D 能力落地确认"（构造 + 数值 smoke test，固定版本），确认可用则归主对照，否则降级为 `future-xfdtd` 并记录判定日期与 Tidy3D 版本。

### 步骤 S0 — 冻结与止血（本周内）
- **S0.1** 把所有声明 E2/E3 但实测 E0/E1 的计划状态回退为 `reopened`（01 已回退；03/04/06/07 在其完成记录追加"证据级实测"栏）。
- **S0.2** 冻结 Wave C/D 新物理实现：05/08/09 的求解器消费、10–13 一律不启动，直到 S3 通过。
- **S0.3** 建立"门分类"规范文档：每个数值门必须自标 `analytic-identity | tautology | symmetric | postprocess-only | wave-level`，禁止用前四类作为 exit gate 的唯一证据。

### 步骤 S1 — RF 端口 wave 级验证（P0，阻塞 03/04/06）
进入门：S0 完成。目标证据：E2。
- **S1.1** 建 `benchmark/scenes/rf/`：`lumped_open_short_match`、`series_parallel_rlc`、`coax_thru`、`microstrip_two_port`、`rectangular_waveguide`（01 §10 前 6 项）。每个跑**真实 FDTD**，输出 S/Z 与解析传输线比较，并按参考策略叠加 **Tidy3D 对照**（这些能力 Tidy3D 均覆盖，经现有 benchmark adapter 生成缓存参考），落 RESULTS。
- **S1.2** 替换 01 Phase 1/2 的恒等式门：wave 级 matched |S11|、FDTD RLC 谐振（真跑）、**非对称**二端口互易 + 物理功率平衡。
- **S1.3** 网格/时间/运行长度收敛报告（三档），每场景附守恒/无源性检查。
- 退出门：上述场景达到 01 §10 阈值且为 wave-level；否则按实测阈值如实记录并标注差距，不得再以 API 存在充数。

### 步骤 S2 — 端口热路径性能（P0，与 S1 并行）
进入门：S0 完成。目标证据：E2。
- **S2.1** 复现 14.4× profile 作为基线工件（脚本入 `tests/rf/performance/`）。
- **S2.2** 实施审计给出的三选一修复：融合标量电路更新为单 kernel / 整步入 CUDA graph / 给端口预置 GPU DFT 权重表（复用 `build_dft_step_tables`）；能量诊断改 opt-in（已部分完成，需确认默认关闭路径无回退）。
- **S2.3** 达成 §9.4 目标：单 LumpedPort+181 freq 步进开销 <5%，每增被动端口 <2%，用 CUDA event + 预热 + 多重复中位数 + 方差报告。
- 退出门：性能门改为方差感知判据（如 95% CI 上界 < 目标），不再单点 min。

### 步骤 S3 — 网络/协同/阵列的 E2 证据（P1）
进入门：S1 通过（03/04/06 继承的端口功率约定已 wave 级可信）。目标证据：E2。
- **S3.1** 03：把 Phase 4 gate (d) 从网格限定升为通用；主对照为独立 S 参数级联工具，并先做 Tidy3D lumped/network 能力落地确认（🟡）——可用则叠加 Tidy3D，否则标 `future-xfdtd`。
- **S3.2** 04：加多场景守恒/能量残差 + 独立电路求解器（离线参考）交叉验证。EM+电路瞬态强耦合 Tidy3D 不覆盖，端到端强耦合门标 `reference: future-xfdtd`。
- **S3.3** 06：完成 `scene_gradient_vjp` 聚合 per-column 伴随（依赖 02 Phase 7 result-aggregation），关闭 fail-closed；在独占窗口跑 96³/4096-step 资格化。

### 步骤 S4 — 多 GPU 收敛（P1，可与 S3 并行）
进入门：S0 完成。目标证据：前向 E3 / 伴随 E2。
- **S4.1** 在独占计时窗口补齐 ensemble 与 joint-solve 的**计时/加速比/利用率**工件（当前全 deferred）。
- **S4.2** 关闭 `DistributedFDTD` trainable guard 缺口；实现分布式伴随反传或明确将其降级为 non-goal 并文档化。
- **S4.3** 逐项评估 §1.2 的 guard 拒绝清单：哪些是永久物理限制、哪些是待实现，分别标注。

### 步骤 S5 — 基准全量收敛（P1，横切）
进入门：S1 完成。目标证据：E2。
- **S5.1** 逐一诊断 §1.7 超标场景（multi_dielectric / custom_current / lorentz / modulated / high_eps flux），定位是参考 stale、网格不足还是实现误差，如实记录并修复或降级标注。这些均为 Tidy3D 覆盖的物理，参考须重跑并固定 Tidy3D 版本（排除 stale 缓存这一常见混淆源）。
- **S5.2** 关闭 TFSF/衍射绝对 R/T/A 归一化（两份 gap 文档的公开欠账）。
- **S5.3** 把 S1 的 RF 场景纳入 RESULTS 常驻。

### 步骤 S6 — Wave C 求解器消费（P2，S1–S3 全过后方可解冻）
- 07 有损 wire 电流递推 + `ohmic_loss` + 电导率伴随；08 ferrite 编译器/kernel slice；09 SIBC 运行时消费。每项独立走 E0→E2，且必须带独立参考。按参考策略：**09 主对照 Tidy3D**（SIBC + Hammerstad/Huray）+ Leontovich 解析；**07/08 Tidy3D 不覆盖**，先以解析占位（07 解析趋肤、08 gyrotropic 谐振/Faraday 旋转解析）并标 `reference: future-xfdtd`。

### 步骤 S7 — Wave D（P3，按产品线选择性启动）
- 10–13 仅在明确产品目标后按 gap 文档 P2/P3 启动，且必须复用 01 的 `PowerLossData` 契约，不新建重复数据模型。

## 4. 勾销的硬性定义（防止再次证据通胀）

一个计划 phase 只有同时满足以下，才能标 `completed`：
1. exit gate 声明的证据级有**实测工件**（非 API 存在、非代数恒等式、非同义反复、非对称 fixture、非后处理喂解析）。
2. 至少一类**独立参考**（解析解 / 独立求解器 / 标准数据 / 守恒律），且外部求解器对照遵循 §3 "S·参考求解器策略"：Tidy3D 覆盖的能力必须用 Tidy3D 对照，未覆盖的标 `reference: future-xfdtd` 并以解析/独立求解器占位（不得因此跳过或自证）。
3. 涉及 PDE/几何离散的能力附**收敛报告**。
4. 相关 benchmark 场景**存在且进入 RESULTS**，当前值达标或差距如实标注。
5. 由计划作者以外的复核（本审计模式）确认后，方可从 `reopened`/`active` 转 `completed`。

## 5. 附：本次变更

- `docs/archive/plans/01-rf-engineering-workflow.md` → `docs/plans/next-functional-2026-07/01-rf-engineering-workflow.md`（git mv，保留历史），状态由 `completed` 改为 `reopened`，加复核结论指向本文。
- 本审计报告：`docs/assessments/next-functional-audit-2026-07-18.md`。
