# RFIC 分层介质 MoM 求解器立项计划

> 状态：**已立项（用户 2026-11-06 批准，"全部按建议"）**。MoM 线就本计划恢复，第 10 节决策全部按推荐项冻结；v4 Stop B 仍暂停
> 日期：2026-11-06
> 类型：新 formulation、新 Green 函数、新 kernel 体系。不是现有自由空间 MoM 的参数扩展
> Owner：由协调者指派（实现：Codex worktree；规格、评审、GPU 验收、合并：Claude 线 owner）
> 关联：`mom_open_surface_thin_conductor.md`（Phase 5/6/8 的内容由本计划接管）、`mom_provider_epoch_v4_acceleration.md`（保持暂停，不是本计划的前置）、`<workspace>/numerical_equivalence_standard.md`（约束）

## 0. 一句话

做一个面向片上无源器件（电感、变压器、传输线、MIM/MOM 电容、巴伦、焊盘）的**分层介质混合位 EFIE（MPIE）MoM**：输入 GDS 和工艺栈，输出多端口 S/Y/Z（Touchstone）和 L/Q/R/C。**所有热路径从第一个 commit 起就是原生 CUDA kernel**，不经过"先写 PyTorch 再改 CUDA"的阶段。

## 1. 现状与差距

现有 MoM 是**均匀背景（真空）下的 3D 表面积分方程散射求解器**：闭合介质体用 PMCHWT，开放 PEC 片用 EFIE（带 Calderón 预条件）。后端有 dense、H/H2、FMM、pFFT，全部用 PyTorch 实现。它和 RFIC 需求的差距如下：

| 需求 | 现状 | 证据 |
|---|---|---|
| 分层介质 Green 函数（Sommerfeld、表格化） | 无 | `witwin/maxwell/mom/` 中没有 layered/Sommerfeld；`physics.py:220` 只接受无界均匀外部 |
| MPIE（矢量位和标量位分开） | 部分（仅自由空间） | `single_layer.py` 的矢量位块和 P0 标量位块分开装配 |
| 过孔、junction | 无 | `efie_open.py:163` 拒绝相交的片 |
| 导体损耗（SIBC、有限厚度、粗糙度） | 无 | `topology.py:200` 拒绝 `LossyMetalMedium` |
| 极低频稳定（λ/1e3 到 DC） | 部分：Calderón 只验证到 0.01λ，准 Helmholtz 投影失败，未交付 | `mom_open_surface_thin_conductor.md` Phase 3 记录 |
| 端口、S 参数 | 无 | `physics.py:216` 拒绝 `Scene.ports/lumped_elements/circuits` |
| 扫频、无源性 | 无（MoM 侧） | 共享模块 `rational.py` 已有有理拟合和无源性检查 |
| GDS、工艺栈前端 | 部分 | `adapters/gds.py` 能把 GDS 导入成拉伸多边形；没有工艺栈，也没有版图网格器 |
| 原生 GPU kernel | 无：MoM 全部是 PyTorch | `witwin/maxwell/mom/` 中没有 `.cu` 文件 |

可以复用的部分：RWG 和拓扑分类（`topology.py`、`open_surface.py`）；Duffy 和 gap-aware 积分规则（作为**算法设计**和 **test oracle** 复用，代码不移植）；pFFT 网格设计经验；共享的 `LumpedPort`/`TerminalPort`（`ports.py:164`、`:305`）；`touchstone.py`；`rational.py`；`roughness.py`（Hammerstad/Huray/Groisse）；`LossyMetalMedium`（`media.py:2244`）；FDTD PCB 线（同一个 Scene 可以交叉验证）。

## 2. 目标与非目标

**目标**
- 器件：螺旋电感、差分电感、变压器、微带和 CPW 传输线、MIM/MOM 电容、巴伦、焊盘和 RF 走线，可以带图形化地屏蔽层。
- 工艺：任意层数的平面分层介质（典型 BEOL 有 10–20 层，包括有耗硅衬底、多层厚金属、过孔层），可以带顶部和底部的 PEC 地或开放边界。
- 频段：从接近 DC（与 DC 电阻连续）到 mm-wave。具体上限在 D2 决定，暂定 ≤ 200 GHz。
- 输出：多端口 S/Y/Z（Touchstone），差分和共模，电感的 L(f)、Q(f)、R(f)，自谐振频率；电流和近场只作诊断用。
- 精度目标（D2 冻结）：L 误差 ≤ 1–2%，Q 峰值误差 ≤ 5%，|S| 误差 ≤ 0.01，相位误差 ≤ 1°。都对照第 8 节的参考。

**非目标（v1）**
- 有源器件、非线性、热、时域瞬态（时域行为通过有理模型得到）。
- 非平面任意 3D 封装（键合线、BGA）：放到 v2，届时接入现有自由空间 MoM 或 FDTD。
- 几何梯度和逆设计：v1 只预留伴随接口，不交付。
- 不修改现有自由空间 MoM 的数值行为，不触发 v3/v4 provider epoch（见第 9 节）。

## 3. 原生 GPU kernel 原则（本计划的硬约束）

延续用户 2026-11-03 的决定：加速路径是唯一路径，新功能 CUDA kernel 优先。对本计划具体化为以下规则：

1. **没有 PyTorch 原型阶段。** 所有计算密集的步骤都直接写成 CUDA kernel：Sommerfeld 积分、Green 表构建和插值、奇异/近场/远场积分、矩阵装配、端口 RHS、loop-tree 变换、pFFT 投影和插值、预校正、GMRES 内核。走 `torch.utils.cpp_extension`，构建方式和 `fdtd/cuda/build.py`、`compiler/accel/build.py` 一致。
2. **PyTorch 的允许用途只有三种：** (a) 编排代码和张量容器，即调度、形状记账、host 元数据；(b) 调用厂商库：cuFFT（`torch.fft`）、cuBLAS、cuSOLVER（`torch.linalg`），这些本身就是原生实现；(c) **独立的 test oracle**。
3. **Oracle 必须独立实现。** Oracle 是 CPU fp64 实现，跑小规模，并且尽量用**不同的算法**，例如 Sommerfeld oracle 用实轴加极点提取的自适应积分，GPU 用变形路径加表格化。Oracle 不是 kernel 的逐行翻译，也不进入 runtime。
4. **hot path 里禁止：** 对面对、频点、端口、层对做 Python 循环；每个阶段只允许一次（或常数次）launch；求解循环里不能有 host-device 同步，也不能有 `.item()`。
5. **确定性**（数值等价标准规则 1）：默认路径不对共享累加器做浮点 atomic。Galerkin 装配用"面对 tile 计算 → tile 内缓冲 → 按固定顺序 gather 到边对"，或者以边对为中心重算；R0 阶段测量后二选一。重复运行必须 `torch.equal`，每个门至少跑 3 次。
6. **等价性**（规则 2）：CUDA 对 oracle 用导出界判定，不要求 `torch.equal`。离散量（拓扑分类、面对分类、准入决定、端口边集合）必须完全相等。
7. **性能验收以 kernel 为单位：** 每个 kernel 合并前附 Nsight Compute 报告，主导资源（带宽或 FLOP）达到 roofline 的 **≥ 50%**，具体阈值 R0 阶段按 kernel 冻结。端到端时间按第 7 节的标准 fixture 计。
8. **架构：** 本地发 sm_86 SASS，同时带 PTX；云上目标 sm_90，按 `cloud/docs/FACTS.md` 预编译。
9. **精度策略是设计的一部分**（D5）。A6000（GA102）的 FP64 吞吐只有 FP32 的 **1/64**（约 0.6 TFLOPS），所以：
   - 奇异提取、Green 表尾部、低频 loop-tree 缩放和累加用 fp64，或用 double-float/补偿求和；
   - 光滑的远场相互作用和 pFFT 网格用 complex64，误差计入预算；
   - dense 求解默认用 **complex64 LU + fp64 迭代精化**，精化不收敛就明确失败；
   - 需要大规模 fp64 dense LU 时上云用 H100（FP64 tensor core），每次按 `wcloud` 流程审批。
10. **与 core-CUDA 线共同设计**：kernel 接口、融合方式、构建和 ABI 约定，要和 core-CUDA owner 一起评审。

## 4. 数值方案

- **Formulation：** Michalski–Zheng Formulation C 的 MPIE。未知量是 RWG 面电流，包括竖直面上的电流；Galerkin 测试。电荷由 RWG 散度给出。
- **分层 Green 函数：** 先用传输线等效求谱域 Green 函数，layer 递推每个 kρ 一个线程。空间域用 Sommerfeld 积分，取变形积分路径，并做准静态项和表面波极点提取。结果按 (ρ, z, z′) 建表并在 kernel 内插值。DCIM 只作为对比项，不作为默认实现，因为它的鲁棒性问题已经广为人知。
- **奇异性：** 自由空间奇异部分加准静态镜像项用解析或 Duffy 处理；剩下的光滑部分用表格插值，正则积分。
- **过孔：** 竖直面电流加跨层的 Green 分量（G_zz、G_zx，以及 Formulation C 的标量位修正）；带 junction 拓扑（T 形、片–过孔连接）。
- **导体：**
  - **v1**：薄片加有限厚度表面阻抗 Z_s(ω, t)，在 DC 极限退化为 1/(σt)；粗糙度复用 `roughness.py`。
  - **v1.5**：厚金属用"上下表面 + 侧壁"的闭合 SIBC 模型，以捕捉电流拥挤和侧壁电流。
  - 体电流（PEEC 类）模型作为备选，D4 决定。
- **低频：** loop-tree（tree-cotree）分解加频率缩放，并显式处理 DC 极限。树的生成在 GPU 上做，用确定性的 BFS 或 union-find。
- **端口：** 在端口边集合上加 delta-gap 电压源；端口语义复用 `LumpedPort`/`TerminalPort`，让同一个 Scene 也能在 FDTD 上跑。支持差分端口、参考阻抗和端口寄生去嵌入（SOC 或校准结构法，R2 选定）。所有端口一次因式分解、批量求解多个 RHS。
- **求解：**
  - ≤ 约 40k 未知量用 dense LU。complex128 时 40k 未知量的矩阵占 25.6 GB；complex64 减半。
  - 更大的问题用 GMRES 加近场稀疏预条件，算子用分层 pFFT。分层 kernel 只在横向平移不变，所以每个层对做横向 2D FFT，z 方向直接处理。
- **扫频：** 自适应频率采样加有理拟合和无源性检查/强制，复用 `rational.py`。几何量与频率无关的部分在各频点之间复用；Green 表按频率批量构建。

## 5. Kernel 清单

| ID | Kernel | 精度 | 确定性要点 | Oracle |
|---|---|---|---|---|
| K1 | 谱域 TL Green 函数（layer 递推，按 kρ×频率×层对批量） | fp64 | 无归约 | CPU fp64 递推，以及解析的单层、半空间 |
| K2 | Sommerfeld 积分加建表（变形路径、极点/准静态提取） | fp64 | 每个表节点内固定顺序求和 | CPU 实轴自适应积分加极点留数 |
| K3 | 表格插值（融合进 K4/K5） | fp32 或 fp64（D5） | 无归约 | 直接调用 K2 逐点求值 |
| K4 | MPIE 近场/奇异装配（Duffy + 镜像项 + 插值） | fp64 | tile 缓冲 + 固定顺序 gather | CPU fp64 暴力积分；均匀栈极限下对比现有 `efie_open` |
| K5 | MPIE 远场正则装配 | fp32 或 fp64 | 同 K4 | 同 K4 |
| K6 | 竖直电流和过孔的跨层项 | fp64 | 同 K4 | CPU 实现 |
| K7 | 表面阻抗 Gram 项和端口 RHS、V/I 提取 | fp64 | 固定顺序 | 解析值 |
| K8 | loop-tree 生成和基变换 | 离散量精确；数值 fp64 | 确定性树 | CPU 图算法（离散量必须完全相等） |
| K9 | 分层 pFFT：投影/插值（排序后分段归约）、横向 FFT 卷积、稀疏预校正 | complex64 + fp64 预校正 | 不用 atomic | 与 dense 比较 |
| K10 | GMRES 融合内核（正交化、真实残差） | 与算子精度一致 | 固定归约树 | CPU GMRES |

dense LU、GEMM、FFT 调厂商库（cuSOLVER、cuBLAS、cuFFT），不自己写。

## 6. 分阶段计划

每个阶段都要有显式的拒绝路径：不支持的组合直接报错，不能静默运行。新 formulation 的 identity 是 `mpie_layered_v1`。

| 阶段 | 内容 | 验收 | 工时（Codex+评审） / GPU |
|---|---|---|---|
| **R0** 规格冻结 | 工艺栈 schema（`LayerStack`）、Scene 和端口 API、精度策略、kernel 接口和 ABI、装配确定性方案、各 kernel 的 roofline 目标、oracle 清单、fixture 清单 | 用户批准第 10 节的决策；与 core-CUDA owner 评审 kernel 接口 | 2–4 h / 0 |
| **R1** 分层 Green 函数 | K1–K3 | 对 oracle 随机点比较，相对误差 ≤ 1e-8（fp64）；均匀栈与自由空间完全一致到 1e-12；PEC 地与镜像法完全一致；互易性 G(r,r′)=G(r′,r)；插值误差在准入范围内 ≤ 1e-6；有耗衬底和表面波极点的算例；性能：15 层栈 × 50 频点的建表时间 | 6–10 h / 2–4 |
| **R2** 平面 MPIE + 端口 + S 参数（dense） | K4、K5、K7（端口）、dense 求解、`Result.network`、Touchstone 输出 | 均匀栈极限与现有 `efie_open_pec` 一致；微带 Z0 和 ε_eff 对照 Hammerstad–Jensen 及 FDTD PCB 线；λ/2 短截线谐振；无损互易 S12=S21 到 1e-10；无损时 S 酉性 1e-8；确定性 3 次重复 | 8–14 h / 3–6 |
| **R3** 过孔和 junction | K6、junction 拓扑、多层连接 | junction 处 KCL 残差；过孔电感对照解析近似和 FDTD；两层叠层电感对照 FDTD | 6–10 h / 2–4 |
| **R4** 导体损耗 | 有限厚度 Z_s、粗糙度、厚金属的闭合 SIBC（v1.5） | 低频 R 与 ρL/(Wt) 一致；趋肤效应过渡对照二维截面参考；Q(f) 对照 FDTD 和参考数据 | 6–10 h / 2–4 |
| **R5** 低频稳定 | K8、频率缩放、DC 极限 | 100 µm 电感从 1 MHz 扫到上限，条件数有界；L(f) 平坦到 DC；R(f) 连续到 DC；迭代次数与频率无关 | 6–10 h / 2–4 |
| **R6** 有耗硅衬底和屏蔽层 | 衬底损耗、图形化地屏蔽层场景 | 10 Ω·cm 衬底上电感的 Q 峰值和自谐振频率对照参考；有无屏蔽层的 Q 差异方向正确，且量级可以解释 | 3–6 h / 1–3 |
| **R7** 生产前端 | GDS 加工艺栈导入；Manhattan 版图网格器（棱边渐变加密，避免 sliver）；自适应扫频加无源性 | GDS 往返；扫频点数 ≤ 均匀扫频的 1/5，且误差在目标内；无源性检查通过；网格质量门（避开已知的 sliver 近场积分缺陷） | 8–12 h / 2–4 |
| **R8** 大版图 | K9、K10、近场预条件 | 在与 dense 重叠的规模上相对误差 ≤ 1e-4；规模序列到约 1e6 未知量（云端，按次审批）；内存和时间报告 | 10–16 h / 本地 4–8 + 云 H100 若干小时 |
| **R9** 认证 | 参考集战役，冻结 provider identity | 第 8 节全部参考通过第 2 节的精度目标；打包证据，独立回读 | 4–8 h / 6–10 |

**里程碑**
- **M1**（R0–R2）：无损平面分层 S 参数。
- **M2**（加 R3–R5）：**第一个可用的 RFIC 版本**，能给出宽频电感和变压器的 L/Q。
- **M3**（加 R6–R7）：完整的 GDS → Touchstone 流程。
- **M4**（加 R8–R9）：支持大版图，并通过认证。

**总量（粗估）：** 工程约 60–100 h，GPU 约 25–45 GPU-h（本地），加上 R8 的云端规模测试。实际瓶颈是 GPU 时间、评审和用户决策，不是写代码。R1 完成后，R2 和 R4 的 Z_s 部分可以并行；R5 可以和 R3 并行。

## 7. 标准性能 fixture

R0 冻结以下 fixture 和时间预算（A6000，单卡）：
- F-A：3.5 圈方形螺旋电感，2 层金属加过孔加衬底，约 5–10k 未知量。目标：单频点装配加求解在秒级。
- F-B：变压器，4 端口，约 20k 未知量。目标：dense 模式下单频点 ≤ 1 min。
- F-C：大版图，包括多器件和走线，≥ 2e5 未知量，走 pFFT。目标在 R8 前由规模序列确定。

每个 fixture 都记录：未知量数、建表/装配/因式分解/求解各阶段的时间、峰值显存、kernel 级 roofline 达成率。

## 8. 验证参考

1. **解析解：** 微带和带状线的闭式公式、镜像法、DC 电阻、过孔电感近似、均匀栈退化为自由空间。
2. **仓库内部交叉：** 同一 Scene 跑 FDTD PCB 线（端口复用 `LumpedPort`/`TerminalPort`）；均匀栈极限对比现有自由空间 `efie_open`。
3. **独立 oracle：** 第 5 节的 CPU fp64 实现。
4. **外部参考（D3：推荐，作为首选参考）：** 是否能用外部参考求解器（商业分层介质 MoM 或 3D FEM 工具，包括受限的免费版本）、已发表的电感测量数据，或开源 FDTD。没有外部参考时，M2/M4 的结论要写明只基于内部交叉验证。

## 9. 边界与流程约束

- **新代码放在新模块**（如 `witwin/maxwell/mom/layered/`、`witwin/maxwell/mom/cuda/`）。现有 MoM 文件只在分发处加一行路由。改动现有冻结文件之前，先和 Bench owner 确认是否触发 provider epoch。
- 现有 PyTorch MoM 不在本计划里移植，也不删除；它在本计划中只作为均匀栈极限的 oracle。
- **GPU：** cuda:1 优先给 core-CUDA，cuda:0 归 coverage 线（需要申请，且空闲 ≥ 10 GiB）。本计划的 GPU 时段由协调者分配。云端只能通过 `cloud/bin/wcloud`，每次 launch 由用户批准该次计划。上传内容只有代码和自己生成的数据，不上传 PDF/TeX。
- **测试：** 每个改动跑快速测试；每批跑一次完整回归；重型战役只在里程碑处跑（批量验证策略）。
- **Codex brief：** 写明不做防御性编程，测试只跑指定的验收集，diff 要小；GPU 测试在主会话执行。

## 10. 用户决策（2026-11-06 冻结）

用户原话："全部按建议，commit 并交给协调者  可以使用外部参考，甚至推荐使用外部参考。"

- **D1 批准。** 恢复 MoM，单独立这条线。v4 Stop B 继续暂停，两者独立。
- **D2 批准暂定值。** 频段上限 ≤ 200 GHz；L 误差 ≤ 1–2%，Q 峰值误差 ≤ 5%，|S| 误差 ≤ 0.01，相位误差 ≤ 1°。R0 按选定的参考把每个 fixture 的阈值冻结下来。
- **D3 推荐使用外部参考。** 外部参考求解器是 M2/M4 结论的首选参考，内部交叉验证（FDTD、oracle、解析解）作为补充，不能替代外部参考。R0 要盘点可用的外部参考：工具及许可、已发表的测量数据，并为 F-A/F-B 等 fixture 指定参考。
  - 外部求解器的产品名只写在 adapter 或参考缓存的专用文件里，遵守仓库命名规则。
  - 需要付费额度或云端运行的参考，单独估价，每次报用户批准。
- **D4 批准表面模型优先。** v1 用有限厚度 Z_s，v1.5 用厚金属闭合 SIBC。体电流模型只在 R4 用参考证明表面模型达不到 Q 目标时才另立项。
- **D5 批准混合精度。** 本地默认 complex64 LU + fp64 迭代精化（奇异项、低频部分用 fp64）；大规模 fp64 dense 上云 H100，每次按 `wcloud` 流程审批。
- **D6** 交给协调者指派 owner、排 GPU 时段。

### 原待决事项（已按上面决策关闭）

- **D1** 恢复 MoM 并立项本线。v4 的 Stop B 继续暂停，不影响本线，因为本线不依赖现有 PyTorch 快速后端。
- **D2** 目标器件、频段上限和精度目标（第 2 节的暂定值）。
- **D3** 外部参考：有没有商业工具许可或测量数据？
- **D4** 厚金属模型：先用表面 SIBC（推荐，速度快），还是直接做体电流（电流拥挤更准，但未知量多数倍）？
- **D5** 精度策略：本地默认用混合精度（complex64 LU + fp64 精化），大规模 fp64 上云 H100（推荐）；或者全程 fp64，接受 A6000 上 fp64 吞吐 1/64 的代价。
- **D6** 由协调者指派 owner，并确定 GPU 时段。
