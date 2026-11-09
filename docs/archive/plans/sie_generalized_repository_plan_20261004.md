# Maxwell 广义积分方程求解器：完整实现、鲁棒性与生产验收计划

状态：Goal 活跃；GSP-00 已完成；GSP-01 双卡继任回归及独立审计通过，物理精度尚未达标
编制时间：2026-09-20 11:42 PDT
当前代码基线：`witwin-maxwell@068c60587aba24cb882e3921ee05988dd812bebf`

## 0. 新 Session 应创建的 Goal

在新 Session 中读取本文件后创建以下 Goal，不要继续使用旧的 paused Goal：

> 完整实现并验收 Maxwell 广义积分方程求解器：先在当前 HEAD 完成现有 SIE 范围的产品化与鲁棒性继任验收；完整支持多个互不接触、互不相交、互不嵌套的闭合物体；完整支持公开材料契约内的磁性、各向异性与非线性材料；完整支持版本化的 1D/2D/3D 准周期 Green 函数；实现可配置的 GPU H/H²-matrix 与自由空间/周期快速 FMM 路径；同步公共 Scene→Simulation→Result API、Bench domain randomization、精确存储、证据和开发计划。开表面、多层/嵌套介质、内部源和几何梯度明确不在本 Goal。每个大阶段独立提交，最后运行全量测试、物理/复杂度验收和独立子代理审计。

旧线程的 Goal `01a091e6-6804-7571-89b6-cc045348ed97` 仍为 paused 且未完成，所以当前线程的 Goal 工具拒绝直接替换。新 Session 应直接用上面的 objective 创建新 Goal；只有工具仍报告继承了未完成 Goal 时，才先结束/替换旧 Goal。

## 1. 结论与范围决策

本轮应完成的是一个版本化的广义积分方程求解器族，而不是把所有路径错误地统称为同一个标量 PMCHWT SIE：

| `formulation_id` | 数学路径 | 用途 |
| --- | --- | --- |
| `pmchwt_scalar_sie` | 标量各向同性区域 Green、表面 J/M | 当前单体、多物体、各向同性磁材料 |
| `anisotropic_dgf_sie` | 均匀张量介质的 dyadic Green、表面 J/M | 线性各向异性与张量磁材料 |
| `coupled_sie_vie_linear` | 表面 J/M + 线性体极化未知量 | SIE–VIE 线性基础与退化 oracle |
| `coupled_sie_vie_hb` | 表面 J/M + 体极化未知量 + harmonic balance | bulk χ²、χ³、Kerr、TPA 稳态非线性 |

周期性与快速算法是可组合轴，不是互斥 formulation：

- `kernel_id = free_space_helmholtz | quasiperiodic_ewald_1d | quasiperiodic_ewald_2d | quasiperiodic_ewald_3d`，Wood/谱极点路径另带版本化 shifted/limiting 标识；
- `operator_backend = dense | hmatrix | h2 | fmm | pme | auto`；`pme` 是独立的周期网格算法，不能计作 FMM 验收；
- `formulation_id + kernel_id + operator_backend + tolerance/version` 共同组成缓存、Result、snapshot 和 evidence 身份。

物理上必须诚实：一般 bulk 非线性不能只靠表面 J/M 和单频标量 Green 完整描述。它需要体极化未知量与多频耦合，因此该路径必须标记为 SIE–VIE harmonic-balance，而不能宣称“纯表面线性 SIE”。

### 1.1 本 Goal 包含

- 当前单闭合物体 SIE 在新 HEAD 上的完整继任验收和鲁棒性闭环。
- 多个互不接触、互不相交、互不嵌套的闭合均匀物体。
- 每个物体可具有不同的合法材料。
- 均匀各向同性磁性材料，包括频率采样后的 `mu_r`、`sigma_m` 和磁色散。
- 公开材料 API 中可物理定义的 reciprocal full/diagonal ε、μ 张量；gyromagnetic/nonreciprocal μ 单独验收。
- 公开材料 API 中的标量 χ²、χ³/Kerr、TPA 的稳态 harmonic-balance 解。
- 1D/2D 开放方向准周期散射；3D 全周期核与谱极点语义。
- GPU H-matrix、H²-matrix、自由空间 Helmholtz FMM、周期快速路径和统一算子协议。
- forward/adjoint、多 RHS、精确近场、持久化、失败恢复、复杂度和独立物理参考。
- Bench 对对象数、布局、材料张量、非线性参数、晶格/Bloch 参数和响应流形的随机化与防泄漏。

### 1.2 本 Goal 明确不包含

- 开表面、薄片、PEC screen、half-RWG。
- 多层、嵌套界面、包覆颗粒、内部空腔的 region graph。
- 内部点源、内部电流源；外部源扩展也不是本轮必要条件。
- 几何/顶点/半径梯度。
- 任意瞬态或脉冲非线性；这仍属于 FDTD。
- 将未通过组合测试的能力做笛卡尔积式宣传。

## 2. 当前事实与基线

### 2.1 仓库状态

- 当前 Maxwell HEAD：`068c605`。
- 根计划已添加当前 `068c605` 基线及 exact-e95 失败事实；旧 `8c0a8a3` 段落保留历史身份。
- `8c0a8a3..068c605` 的 22 个提交主要新增 FDFD GPU local-response/优化能力；没有实现本计划要求的 SIE 扩展。
- 这些提交修改了共享 `simulation.py`、`result.py`、`compiler/sources.py`，所以历史 SIE 验收不能自动覆盖新 HEAD。
- 首次核查时 Maxwell 工作树干净；最终文档审计时出现未跟踪的 `scripts/reference/measure_pngf_precompute.py`，来源未归因，GSP-00 必须先判断其所有权并保留，不得覆盖或顺手提交。Bench 仍有原有 18 项未跟踪历史证据。

### 2.2 当前 SIE 基线实跑

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  -m pytest -p no:cacheprovider -q tests/sie
```

结果：

- 收集 341 项；
- `123 passed, 217 skipped, 1 failed`；
- 此次历史基线在沙箱内不能初始化 NVML/CUDA；后续已确认沙箱外两张 RTX A6000 可用；
- 唯一失败：`tests/sie/test_single_buffer_assembly.py::test_actual_fresh_entrypoint_preflight_no_cuda`；
- 失败原因为 `Scientific inventory differs from exact e95`，即历史 exact-e95 源清单拒绝当前 `068c605`，不是数值结果通过。

这必须通过新一代继任 epoch 解决，禁止放宽旧 allowlist 或改写历史 receipt。

### 2.3 当前实现边界

- `physics.py` 强制恰好一个结构、标量 ε、`mu_r=1`，拒绝各向异性/非线性/周期。
- `runtime.py`、`SurfaceData`、snapshot 只有一个 mesh/basis/interior。
- `operator.py` 使用自由空间标量 Helmholtz Green，组装 dense `2E×2E` PMCHWT。
- `linalg.py` 大量路径依赖 `.matrix`；实验 streamed action 每次重算全部面片对，仍是 O(N²)，不是 FMM/H-matrix。
- `fields.py` 只有 exterior/interior，周期问题仍错误地沿用孤立体 far field/RCS 语义。

## 3. 全局设计原则

1. 继续使用稳定公共架构 `Scene → Simulation → Result`。
2. GPU-first、PyTorch-native；不得新增生产 CPU 数值 fallback。
3. 先建立 dense oracle，再实现快速后端；快速结果必须对 dense/解析/独立参考验收。
4. 精确近场与奇异积分不能被低秩近似替代。
5. forward 和 adjoint 必须来自同一压缩代数对象。
6. 每个 formulation、kernel、operator backend 和近似容差必须进入结果、snapshot 和证据身份。
7. 不以代数残差代替物理精度，不以小规模压缩代替复杂度证明。
8. 不支持的组合必须在组装前 fail closed，并给出精确原因。
9. 每个大阶段单独 commit；阶段失败证据保留，不重命名为通过。
10. 每个大阶段结束后由未参与实现的子代理进行只读攻击审计。

参考验证采用分层证据，不强制把某一个外部商业软件作为所有阶段的阻塞项：

1. 解析/Mie/对偶/极限解；
2. 当前可信的 Witwin FDTD、dense SIE oracle 与不同离散/算法实现；
3. 网格、积分、谐波、Ewald、rank/order 的独立收敛链；
4. 外部参考求解器只用于其确实支持的物理组合和最终哨兵交叉。

“独立参考”要求数学或实现路径独立，不等同于“必须是外部软件”。如果新物理没有解析解且只有同一代码路径自洽，仍不得据此宣布生产通过。

## 4. 分阶段执行与提交

## GSP-00：合同、基线与计划冻结

目标：把本文件转入仓库版本管理并冻结能力矩阵、公式、误差预算和提交顺序。

工作：

- 在 `docs/plans/` 建立当前活跃计划，历史 SIE 计划保留为快照。
- 更新根计划中的 HEAD 与旧 exact-e95 失败事实。
- 冻结上述四个 `formulation_id`、kernel/backend 正交轴、相量 `exp(-iωt)`、未知量顺序、区域语义、被动性和功率符号。
- 定义 snapshot schema 升级与旧单物体 snapshot 的一次性迁移策略；不保留无界兼容分支。
- 建立能力组合矩阵，禁止单功能通过后自动声称所有组合通过。

验收：文档链接、schema、配置负例和现有测试 collect；独立审计计划无漏项。

建议提交：`Plan generalized integral-equation production expansion`

## GSP-01：当前范围产品化与新 HEAD 继任验收

目标：在 `068c605` 或执行时冻结的新 HEAD 上重新关闭当前单物体标量 SIE。

工作：

- 更新 source inventory，而不是放宽旧 evidence。
- 重跑 `tests/sie`、公共 API、Simulation/Result/source 影响范围。
- 双 CUDA 完成 compact mass、dense/compact Riesz、direct/GMRES/block-GMRES、A/A*、查询和持久化。
- 建立网格坏质量、参数边界、弱散射、损耗、接近共振、缓存失效、OOM 预检、失败恢复和跨设备确定性矩阵。
- 生成新的源绑定 successor epoch，并让 Bench 消费新证据。

验收：无 unresolved failure；所有跳过必须有声明原因；366 节点/1098 阶段或重新冻结后的等价完整清单通过；双设备和源哈希审计 PASS。

建议提交：`Requalify the scalar SIE production baseline`

## GSP-02：统一算子协议与 kernel 解耦

目标：消除求解器对 dense `.matrix` 的硬绑定，为多物体、周期和快速后端建立公共基础。

新增建议：

- `witwin/maxwell/sie/operator_protocol.py`
- `witwin/maxwell/sie/kernels.py`

协议至少包含：

- `shape/device/dtype`
- `apply(x)` / `adjoint(y)`
- Riesz 与 preconditioner action
- `supports_direct`
- backend/kernel/formulation identity
- setup/apply/storage/workspace/counter stats
- deterministic persistence receipt

工作：

- `DenseOperator` 实现该协议。
- GMRES/block-GMRES 完全消费协议；只有 direct/LU 明确要求 dense。
- matrix-free condition diagnostics 改用迭代估计并明确不是 exact SVD。
- `operator.py` 与 `fields.py` 注入 `FreeSpaceHelmholtzKernel`。

验收：自由空间 G/∇G 与旧实现 complex128 达机器精度；dense 全部结果和错误语义等价；dot identity ≤1e-11。

建议提交：`Introduce kernel-independent SIE operators`

## GSP-03：多物体完整支持

### 数学结构

限定多个互不接触、互不相交、互不嵌套的闭合均匀物体。令公共外部介质为 0，物体 i 的内部介质为 i。全局未知量和测试行顺序固定为

```text
x = [J_0, ..., J_(B-1), M_0, ..., M_(B-1)]^T
rows = [H-tested rows for all bodies, E-tested rows for all bodies]
```

对测试物体 i、源物体 j，PMCHWT 子块固定为

```text
A_ij = [ K^0_ij + delta_ij K^i_ii,
         i*omega*(epsilon0*epsilon_r,0*D^0_ij
                  + delta_ij*epsilon0*epsilon_r,i*D^i_ii);
        -i*omega*(mu0*mu_r,0*D^0_ij
                  + delta_ij*mu0*mu_r,i*D^i_ii),
         K^0_ij + delta_ij K^i_ii ]

b_i = [-<f_i,H_inc>, +<f_i,E_inc>]^T
```

因此对角块是公共外部贡献加该物体内部贡献，非对角块只有公共外部贡献。改变物体 j 的内部材料只能改变 `A_jj` 的内部项；严禁改变任何 `i != j` 耦合块或其他物体的内部对角贡献。

### 数据模型

在 `mesh.py` 增加：

- `SurfaceBody(structure_index, name, basis, interior, edge_offset, face_offset)`；
- `SurfaceSystem(bodies, total_edges, slices, global maps, device, dtype)`。

每个物体继续由现有 `SurfaceMesh` 独立验证闭合、连通、定向；不得把断开表面拼成一个 mesh 绕过不变量。

全局系数顺序：

```text
[J_body0, J_body1, ..., M_body0, M_body1, ...]
```

### 文件级改动

- `physics.py`：返回公共 exterior 和 interiors 表；名字/顺序/材料身份。
- `runtime.py`：编译全部结构；signature 绑定所有对象。
- `mesh.py`：跨对象相交、接触、重叠、包含/嵌套检查。
- `operator.py`：test/source basis 分离；跨对象 near/far 分类；正确块公式。
- `quadrature.py::classify_pairs`：同物体允许 self/edge/vertex/near/far；跨物体只允许 near/far，绝不比较各自局部 vertex ID；用真实面片距离/尺度触发近间隙自适应积分并记录最小间距和最高阶数。
- `linalg.py`：块对角 RWG Riesz 与 compact mass。
- `fields.py`：外部叠加全部对象；内部只用目标对象；正确 trace jump。
- `observables.py`：逐对象/总吸收，整体散射/消光。
- `modes.py`：全局 RHS、包围全部对象的控制面。
- `result.py`/snapshot：对象表、offset、逐对象 current、材料与区域身份。

区域语义：

- `0` / `"exterior"`：公共外部；
- `1..B` 按启用结构在 Scene 中的声明顺序绑定；唯一且非空的 Structure.name 可作为别名；`"exterior"`、`"interior"` 是保留名；
- `"interior"` 只允许单物体，多物体时拒绝歧义。

公开数据与查询合同：

- `SurfaceData.J/M` 保持全局拼接向量；`SurfaceData.bodies` 按 Scene 启用结构顺序返回只读视图，每项含 `region_id/structure_index/name/mesh/J/M/global_edge_slice`；`SurfaceData.mesh` 只在单物体有效，多物体明确拒绝歧义。
- `trace_side` 仍只接受 `exterior|interior`。表面查询先确定所属物体，region 必须匹配；外侧迹累加全部物体的外部势，但半跳跃项只来自所在物体；内侧迹只使用该物体的内部核、电流和半跳跃项。
- `PointMonitor` 同时记录 `region_id`、`structure_index` 和可选 `structure_name`；多物体 `"interior"`、表面归属不唯一或一点落入多个内部区域均 fail closed。

多物体 Riesz 固定为 `G=blockdiag(G_0,...,G_(B-1))`、`R=diag(G/|eta_ext|^2,G)`。compact 表示保存每个物体的 Gram/factor 和切片，不得为了块对角结构分配含大量零的全局 E² tensor；inverse、mutation/cache identity、persistence 以及 direct/compact 组合均须验收。

snapshot 至少保存 schema version、全局系数顺序、每个对象的 region/Scene index/name、vertices/faces、edge/face offsets 和 coefficient slices、采样后内部材料、公共外部材料、formulation/kernel/backend、Scene 顺序与完整 signature。对象重排必须拒绝，除非存在显式迁移，不能静默重绑旧系数。

负例：相交、相切、零间隙、嵌套、部分重叠、重复/保留名、查询点多重归属、恢复时对象重排/材料改变全部拒绝。

验收：单物体逐元素退化；逐个 `A_ij` 与独立 test/source quadrature 比较；仅改变物体 j 内部材料时全部非对角块及其他对角块不变；显式注入错误 interior 非对角项时测试必须失败；对象置换满足 `A'=PAP^T,b'=Pb,x'=Px`；无对比多体散射随网格收敛到零；远距离趋于独立解；两球 T-matrix/多重散射独立参考；全局远场等于逐物体复振幅之和且物体 i 内场只由其内部表示解码；近间距 sweep 同时验证自适应阶数、积分、能量与独立参考；互易/能量；direct/GMRES/block、A/A*、complex64/128、保存恢复、双 GPU。

新增测试至少包括 `test_multiobject_mesh.py`、`test_multiobject_operator.py`、`test_multiobject_fields.py`、`test_multiobject_observables.py`、`test_multiobject_persistence.py`、`test_multiobject_public.py`、`test_multiobject_reference.py`。

建议分三次提交：

1. `Add multi-body surface topology and region contracts`
2. `Assemble coupled multi-body PMCHWT systems`
3. `Expose multi-body fields observables and persistence`

## GSP-04：各向同性磁材料

目标：保持纯 PMCHWT SIE，支持均匀线性各向同性磁材料。

工作：

- `compile_medium()` 支持 `mu_r`、`sigma_m`、磁 Debye/Drude/Lorentz 的目标频率采样。
- 正确选择被动 outgoing 分支：`Im(k)>=0`，并用功率方向/`Re(eta)` 处理双负材料，不能继续只看 `Re(k)>0`。
- 移除当前 `Re(epsilon)>0`/`Re(mu)>0` 的硬限制，在被动性条件下覆盖负 ε、负 μ、双负、near-zero 和穿越色散极点的分支连续性。
- 去除 operator/fields/observables/modes 中的无磁硬编码。
- persistence 与 fingerprints 保存完整磁参数和采样结果。

验收：磁介质 Mie TE/TM；ε/μ 电磁对偶；阻抗匹配零散射；σm 衰减；磁极点频扫；无损守恒、有损吸收非负；SIE 与独立体离散交叉；网格/积分收敛。

建议提交：`Support isotropic magnetic PMCHWT media`

## GSP-05：各向异性 dyadic-Green SIE

不能把 ε/μ 张量直接塞进当前 `exp(ikr)/(4πr)` 标量核。

### GSP-05A reciprocal tensor media

新增建议：

- `sie/tensor_medium.py`
- `sie/dyadic_green.py`
- `sie/anisotropic_operator.py`

公共材料契约在本 Goal 扩展为一般 complex passive reciprocal ε/μ 3×3 tensor；reciprocal 要求转置对称，被动性通过与 `exp(-iωt)` 相容的 Hermitian/anti-Hermitian 分解验证。先实现 full ε + diagonal μ，再实现 reciprocal full μ；现有实对称 SPD ε 是其严格子集。谱积分、奇异部分分离、field decoder 和 far/diffraction kernel 必须一致。

验收：张量退化到标量；旋转协变；互易；单轴球/椭球解析或独立参考；被动功率；谱积分与表面积分双收敛；独立 FDTD 对照。

### GSP-05B nonreciprocal magnetic tensors

- 扩展公共 permeability tensor/gyromagnetic API，只接受定义明确、可验证被动性的 complex tensor。
- 左/右圆偏振分裂、偏置反转、非互易散射矩阵、损耗非负和独立参考分别验收。

建议提交：

1. `Add reciprocal anisotropic dyadic Green operators`
2. `Support nonreciprocal magnetic tensor media`

## GSP-06：非线性 SIE–VIE harmonic balance

“完整”限定为公开材料契约内的稳态周期解；任意瞬态仍由 FDTD 负责。

### GSP-06A 线性 SIE–VIE 基础

新增建议：

- `sie/volume_mesh.py`
- `sie/volume_basis.py`
- `sie/vie_operator.py`

先用线性体极化与表面 J/M 联立，对均匀标量物体退化到 PMCHWT oracle。Result 和 evidence 必须标记 `coupled_sie_vie_linear`。

材料分解必须唯一：均匀线性背景由各区域 Green 与 SIE 表示；VIE 只承载非线性极化和非线性耗散源，不得再次加入同一线性介质对比。用零体源严格退化和故意双计数的 falsification 测试冻结该约定。

### GSP-06B harmonic balance

新增建议：

- `sie/nonlinear.py`
- `sie/harmonic_balance.py`

`SIEConfig` 增加：谐波集合、体网格、Newton/Picard、阻尼、continuation/pump ramp、分支选择、残差/功率门、最大迭代、谐波截断策略。Result 携带实际频轴、各谐波表面/体未知量、收敛、分支身份、多稳态和功率日志。

公开非线性本构与 FDTD 保持一致并冻结相量归一、谐波卷积和 TPA 耗散符号：

```text
P_i^(2) = epsilon0 * chi2 * E_i^2
P_i^(3) = epsilon0 * chi3 * |E|^2 * E_i
sigma_NL = tpa_scale * |E|^2
```

χ² 必须包含 n=0 optical-rectification 静电块、正/负谐波的实场共轭约束、DC/2ω/3ω 功率账本和谐波截断收敛；相同公开 Material 在 SIE–VIE 与 FDTD 中比较幅值、相位和频谱。continuation 必须稳定追踪指定解支，并记录多稳态而非静默选择。

覆盖公开合法组合：标量 χ²、χ³/Kerr、TPA。当前 API 拒绝的 anisotropy+nonlinearity 组合继续 fail closed，除非先定义张量 χ²/χ³ 模型并独立验收。

验收：

- χ² 振幅二次标度、DC optical rectification、2ω、正负谐波共轭与截断收敛；
- χ³ 三次标度与 3ω；
- Kerr 相移；
- TPA 衰减；
- 零非线性退化；
- Manley–Rowe/总功率；
- Newton/Picard、体网格和谐波阶收敛；
- 与 FDTD 独立交叉。

必须增加 `multi-body + nonlinear` 组合门：每个物体的体未知量只属于该物体，公共外场负责跨物体耦合，线性背景仍不得双计数。若该门未通过，产品 API 必须精确拒绝该组合，不能声称本 Goal 的多体和非线性可组合。

建议提交：

1. `Add coupled surface-volume integral operators`
2. `Implement nonlinear harmonic-balance integral solves`

## GSP-07：准周期 Green 函数

### GSP-07A 1D/2D Bravais 周期散射

新增 `sie/periodic.py`：lattice、reciprocal lattice、Bloch vector、Ewald 参数、G/∇G、误差/截断日志、anomaly guard。

- 零平移 singular 项继续走现有 Duffy。
- 非零像与 smooth remainder 走规则积分。
- 对象第一版不得跨 unit-cell seam。
- 输出 Floquet/Rayleigh diffraction orders；禁止复用孤立物体 RCS/far_field。
- `G(r+R)=phase(R)G(r)` 的相位符号必须按 `exp(-iωt)` 冻结。
- 新增版本化一般 Bravais `LatticeSpec`；一个线性系统及其多 RHS 必须共享同一 Bloch wavevector，入射波平行波矢必须与其模倒格矢一致；不同 Bloch vector 使用不同 operator/cache/signature。
- 传播与倏逝 Rayleigh 阶、grazing 阈值、单位胞功率归一和开放轴 outgoing radiation sign 必须固定；周期近场查询与组装使用同一 periodic kernel。
- 多物体晶胞只有公共 exterior 使用准周期核；每个闭合物体 interior 仍使用该物体的 free-space homogeneous scalar/dyadic self kernel，绝不周期化 interior，也不进入跨体块。
- `periodic + reciprocal anisotropic` 通过新增 `QuasiperiodicDyadicGreenKernel` 实现，包含 dyadic lattice/spectral sum、奇异分离、谱截断、adjoint 与场解码；不得用标量 Ewald 加组合测试冒充实现。

### GSP-07B Wood/Rayleigh anomaly

按周期维数、极化和 kernel 预注册 shifted Green 或 limiting-absorption 的支持域并对极限收敛验收。guided mode、BIC 或真实谱极点仍可能没有唯一有限散射解，必须继续 fail closed；不能承诺一种正则化覆盖全部 Wood 情况。

### GSP-07C 3D 全周期

实现三维准周期 lattice sum、谱极点检测和 limiting absorption。该路径没有孤立散射远场，Result 只能公开单元场、Bloch 通量、谱/模态量。支持域限定为 resolvent 谱外点；真实谱点只返回明确定义的 eigen/residue/projected quantity，否则拒绝。只有 limiting-absorption 极限有限、收敛且路径无歧义时才放行，不能把“正则化”解释为总有有限强迫解。

验收：Bloch shift；Ewald 实/倒格加深；带损耗 direct-image 独立对照；衍射阶功率平衡；互易（适用材料）；外部周期参考；anomaly 与谱极点的正/负例。

建议提交：

1. `Add quasiperiodic Ewald Green kernels`
2. `Resolve periodic anomaly and spectral limits`

## GSP-08：GPU H/H²-matrix 后端

dense 是数值 oracle；先实现非嵌套 H-matrix 作为中间压缩基线，再实现嵌套 H² 基。

新增建议：

- `sie/cluster.py`
- `sie/nearfield.py`
- `sie/hmatrix.py`
- `sie/h2.py`

设计：

- Morton/octree cluster；
- admissibility；
- touching/near blocks 使用现有精确积分，BCSR 保存；
- RWG 源投影为散度和矢量 quadrature moments；
- 远场用 G/∇G 的嵌套 cluster bases；
- tensor DGF 路径实现 tensor-valued kernel traits、dyadic low-rank basis 与误差估计；普通 H² 只承诺 rank 有界的低/中电尺寸，高频使用 directional 分支；
- SIE–VIE 路径实现 volume octree/basis projection、volume singular/near correction、surface-volume cross blocks；不同谐波/频率可共享只读几何树，但 expansion/basis/cache identity 必须分频率保存；
- 测试端再投回 RWG；
- forward/adjoint 共用一个压缩对象；
- batched RHS；
- cache/persistence/OOM preflight。

配置：

```text
operator_backend = dense | hmatrix | h2 | fmm | pme | auto
fast_tolerance
max_rank
leaf_size
admissibility
memory_budget_bytes
```

`auto` 必须在分配前预测容量，不能 OOM 后偷偷回 dense。

验收：

- dense vs fast 多 RHS forward/adjoint；默认 complex128 operator tolerance 建议 1e-8，以阻抗平衡的 test/trial Riesz norm、预注册确定性/随机 probe 和 absolute floor 定义；
- normalized dot-identity defect ≤1e-11；
- 五类 near block逐块一致；
- 统计全部 retained blocks/bases/tree/pair-index 的累计 bytes/count、setup workspace、CUDA peak 增量；禁止 global T² pair list 和 dense matrix，并用多档 storage slope 证明非二次存储，不能只依赖单一 allocator hook；
- 同时报告近似算子自身残差和 dense/original operator spot-check residual；`fast_tolerance` 与 GMRES tolerance 必须纳入统一误差预算；
- J/M、近 E/H、功率和周期衍射误差小于总物理预算的十分之一；
- 分别测固定电尺寸 h-refinement 与固定 points-per-wavelength、随电尺寸增长两种极限。仅在 admissible rank r 随 N 有界时，H-matrix 才以 O(rN logN)、H² 才以 O(rN) 为目标；同时报告核/维度相关 rank 次方、transfer/M2L 常数；
- 报告 crossover，不能只报小网格。

建议提交：

1. `Add exact near-field and hierarchical matrix operators`
2. `Implement GPU H2 surface operators`

## GSP-09：自由空间与周期 FMM

若公共能力使用 “FMM” 名称，必须真正实现 multipole/local expansion，而不能把 H² 重命名为 FMM。

新增建议：

- `sie/fmm.py`
- 对应 CUDA extension/kernels

范围：

- 低/中频 Helmholtz M2M/M2L/L2L；
- expansion order 由 tolerance 与 ka 选择；
- 高电尺寸实现 directional/high-frequency 分支；
- 若 Goal 要求 periodic FMM，必须实现周期 M2L/远像局部展开；PME 可作为独立 `operator_backend=pme` 的 O(N logN) 路径，但不能替代或满足 FMM 验收，H²+Ewald 也不能冒充 periodic FMM；
- 精确 near correction 不重不漏。

验收：与 H² 压缩基线及 dense 数值 oracle 对比、exact adjoint、复杂度和高频 rank/order 增长；预注册 N 序列、几何/ka 尺度族并重复测量。低频 FMM 在固定电尺寸加密下目标 O(N)，directional/high-frequency 通常目标 O(N logN)（依算法与分布），PME 目标 O(N logN)；若经验复杂度未达标，只能称压缩后端，不能称线性复杂度 FMM。

建议提交：

1. `Implement free-space Helmholtz FMM operators`
2. `Add periodic fast multipole acceleration`

## GSP-10：能力组合与公共 API

单项通过不等于组合通过。至少建立以下组合门：

- multi-body + magnetic + dense/H²/FMM；
- multi-body + reciprocal anisotropic + dense/H²；
- periodic + multi-body + magnetic；
- periodic + reciprocal anisotropic；
- nonlinear SIE–VIE + fast volume/surface actions；
- multi-body + nonlinear SIE–VIE/HB；
- nonreciprocal/非线性不满足对称或互易假设时的正确 guard。

公共和持久化更新：

- `runtime.py`, `simulation.py`, `result.py`, `FEATURE_LIST.md`；
- snapshot 保存 formulation、kernel、lattice、对象表、材料采样、tree/near/压缩摘要、误差和性能日志；
- 可重建的树不必默认完整保存，但其构造身份必须可重放；
- legacy snapshot 迁移只保留一个明确版本，不累积兼容分支。

建议提交：`Integrate generalized surface and volume integral workflows`

## GSP-11：Bench randomization、数据和最终验收

扩展 candidate design：

- 对象数与对象族；
- 非重叠位置、姿态、最小间距和耦合强度；
- 每对象材料身份；
- ε/μ 张量特征值、主轴和非互易偏置；
- χ²/χ³/TPA 与泵浦强度；
- lattice vectors、Bloch vector、周期维数和 anomaly 距离；
- backend/tolerance 只作为数值 variant，不进入物理 split identity。

防泄漏：

- scene graph canonicalization 必须消除对象重命名和对象置换别名；
- 对称等价晶胞不能跨 split；
- 同一物理 scene 的 dense/H²/FMM variant 必须在同一 split；
- held-out source、periodic order 和 nonlinear pump 规则单独冻结。

数据顺序：

1. 每个 formulation 的最小 smoke；
2. 32×8 集成 smoke；
3. 4,096-group pilot 与参数/响应 manifold；
4. H2/G0 或相应扩展 gate；
5. 生产 H3；
6. ML05/06 与 MB-06。

最终验收：

- 四仓全量测试；
- 所有 GPU 节点；
- 每个 formulation 的解析/独立参考、网格/积分/谐波/周期截断收敛；
- dense/H²/FMM 精度与复杂度；
- 保存恢复与独立环境重放；
- 失败、OOM、stagnation 和 unsupported 组合；
- 源码/构建/环境/输入/输出/成本双根证据；
- 至少两个独立子代理：数值物理审计、代码/证据/性能审计；
- 更新所有根计划和状态报告。

建议提交：

1. `Extend randomized integral-equation dataset design`
2. `Record generalized solver acceptance and independent audits`

## 5. 验收阈值

阈值应在 GSP-00 冻结；没有预注册阈值不得事后选择最好结果。

最低建议：

- dense/current-scope algebra equivalence：complex128 接近机器精度；
- fast operator forward/adjoint：默认相对误差 ≤1e-8；
- adjoint dot identity：≤1e-11；
- fast 误差：≤总物理误差预算的 1/10；
- 关键场/截面目标：≤0.5%，或保持现有更严格预算；
- 外部参考误差：≤神经目标的 1/4，目标 1/10；
- 能量/功率：按无损/有损/周期/非线性分别冻结绝对和相对底；
- 多对象最小间距、周期 anomaly 距离、材料被动性必须进入适用域；
- H²/H-matrix/FMM 的复杂度必须用预注册 N 序列、固定电尺寸与增长电尺寸两族、重复测量报告经验斜率、rank/order、crossover、retained/setup/peak GPU 内存；不得只用最大三档斜率证明；
- 不满足复杂度门时保持功能可用，但禁止使用 “production fast” 或 “linear-complexity FMM” 声明。

## 6. 测试矩阵

每阶段至少覆盖：

| 维度 | 正例 | 负例/边界 |
| --- | --- | --- |
| dtype/device | complex64/128、GPU0/1、多 RHS | CPU production operand、跨设备对象 |
| geometry | sphere、ellipsoid、harmonic、solid mesh、多对象 | 开面、自交、非流形、接触、嵌套 |
| material | scalar ε/μ、tensor ε/μ、loss、dispersion、χ²/χ³/TPA | 非被动、未定义组合、错误 tensor symmetry |
| kernel | free-space、1D/2D/3D periodic | Wood/spectral pole 未正则化 |
| backend | dense、H、H²、FMM、auto | rank/capacity/tolerance 不可达 |
| solve | direct、GMRES、block、initial guess | stagnation、budget、fallback、tiny RHS |
| query | exterior、逐对象 interior、trace、near/far/diffraction | 歧义 region、周期 RCS、边/顶点 trace |
| nonlinear | zero limit、χ²、χ³、TPA、多谐波 | 不收敛、截断不足、功率不闭合 |
| persistence | save/load、移动目录、重放 | 对象重排、材料/核/树身份篡改 |

## 7. 关键风险与止损条件

1. **范围风险**：这是多个研究级后端，不是一项小扩展。每个 formulation 必须独立 gate。
2. **多体数学风险**：interior kernel 错加到非对角块会产生表面上看似收敛但物理错误的系统。
3. **张量核风险**：各向异性不是标量 Green 乘张量；没有 dyadic singular analysis 就不能发布。
4. **非线性风险**：没有体未知量和谐波收敛就不能声称 bulk nonlinearity。
5. **周期风险**：Wood anomaly 和谱极点需要 shifted/limiting-absorption；否则必须拒绝。
6. **快速后端风险**：H² rank 随电尺寸增长；没有 directional branch 就不能声称通用高频性能。
7. **预条件风险**：FMM 不解决 PMCHWT 低频/稠密网格病态；必要时另立 loop-star/Calderón 阶段，但本 Goal 不实现 mixed Calderón，除非实测成为阻塞。
8. **证据风险**：历史单物体、旧 HEAD 或小矩阵证据不能晋升为新组合资格。
9. **资源止损**：任何阶段若无法在预注册 GPU 内存/时间内形成独立参考和收敛链，保留失败，缩小明确支持域，不能放宽误差门。

## 8. 新 Session 的第一轮动作

1. 创建本文件第 0 节的 Goal。
2. 读取 `witwin-maxwell/AGENTS.md`，确认 `AGENTS.md == CLAUDE.md`。
3. 重新核对四仓 HEAD 和工作树；不得覆盖 Bench 的 18 项历史未跟踪证据。
4. 将本计划复制到 `witwin-maxwell/docs/plans/`，执行 GSP-00 并提交。
5. 为 GSP-01 建立当前 HEAD successor epoch；有双 NVIDIA GPU 才执行真实 CUDA 验收。
6. 在 GSP-02 前不要直接修改 operator 数学；先冻结 dense oracle 与 source inventory。
7. 使用子代理分别审计：多体、材料、周期/快速算法；实现代理不得担任最终审计代理。
8. 每个阶段提交后更新本文件的状态表、测试数、证据路径和 commit。

## 9. 阶段状态表

| 阶段 | 状态 | Commit | 验收 |
| --- | --- | --- | --- |
| GSP-00 合同与基线 | 合同与独立审计完成 | ee5b01a | 341 collect；123 pass / 217 skip / 1 fail（历史 inventory） |
| GSP-01 当前范围继任验收 | 工程回归与独立审计 PASS；物理精度待关闭 | ee14b3f（执行源码） | 521 pass / 1563 phases；双GPU，Bench消费 PASS；无最终物理资格 |
| GSP-02 算子协议/kernel | 待开始 | — | — |
| GSP-03 多物体 | 待开始 | — | — |
| GSP-04 标量磁材料 | 待开始 | — | — |
| GSP-05 张量材料 | 待开始 | — | — |
| GSP-06 非线性 SIE–VIE/HB | 待开始 | — | — |
| GSP-07 周期 Green | 待开始 | — | — |
| GSP-08 H/H² backend | 待开始 | — | — |
| GSP-09 FMM | 待开始 | — | — |
| GSP-10 组合/API | 待开始 | — | — |
| GSP-11 数据与最终验收 | 待开始 | — | — |

## 10. 本计划的独立审计记录

2026-09-20 由三个只读子代理分别审计并在修订后复审：

- 多物体数学、区域/查询、持久化与测试合同：PASS；
- 磁性、各向异性、非线性和 SIE–VIE 材料合同：PASS；
- 周期 Green、H/H²、FMM/PME、复杂度和组合路径：PASS。

这些 PASS 只表示计划已覆盖已知实施与验收缺口，不表示待开发功能已经实现。实现完成后仍须换用未参与实现的代理重新审计代码与证据。

## 11. 参考入口

- 当前 SIE 协议：`witwin-maxwell/docs/reference/sie_protocol.md`
- 当前交付状态：`witwin-maxwell/docs/plans/sie_delivery.md`
- 当前本地验收：`witwin-maxwell/docs/assessments/sie_local_acceptance_closure.md`
- 实验 streamed operator：`witwin-maxwell/benchmark/sie_matrix_free.py`
- Provider candidate design：`maxwell-bench/cases/provider-candidate-design-v1.json`
- 覆盖度量：`maxwell-bench/maxwell_bench/coverage.py`
- 项目群状态：`maxwell_bench_status_report.md`

本文件定义后续开发范围和验收方法，不声明其中任何待实现能力已经完成。

## 12. 执行记录

GSP-00 细化合同见 [sie_generalized_contract_v1.md](sie_generalized_contract_v1.md)。
当前基线和原始日志见 [baseline.json](../assessments/gsp-00/baseline.json)。
已有 341 项收集、123 pass / 217 skip / 1 fail；GSP-00 独立合同复审 PASS。

GSP-01 当前进展与失败记录见 [前检报告](../assessments/gsp-01-preflight/README.md)。

2026-09-20 环境纠正：GPU 不可用仅为沙箱内观察；沙箱外两张 RTX A6000 可用，真实 CUDA 回归已启动。
详见 [CUDA 继任记录](../assessments/gsp-01-cuda-successor/README.md)。

2026-09-20 GSP-01 更新：原始 `81c6a36` 绑定运行实际 517 passed，但工程 grade 因 10 个 host-only 验证节点被误分类为必须分配 CUDA 而失败，原失败记录保留。`ee14b3f` 将这 10 个节点显式标为 host_validation，并要求双卡均零分配；新目录 521 节点 / 239 CUDA 分配节点 / 10 host 节点已获独立审核，正式双卡运行已终态通过：521 tests / 1563 phases，1 条保留 warning；独立终态审计与 Bench 消费 PASS。

当前标量物理探测尚未达到 0.5%：零对比度误差随网格细化下降，但固定网格升高组装/解码积分阶几乎不改变残差。解析入射迹 lifting 仅在仓库外原型验证，其非零弱散射远场误差由 n1 约 12.3% 降至 n2 约 3.28%；这不是生产实现或物理资格。GSP-01 未关闭，后续阶段与完整 Goal 均未完成。

## 2026-10-02 — GSP-09 compact changing-carrier transfer primitive

Explicit compact_phase_v1 factors primary changing-carrier transfers into signed
phase gathers/scatters and separable parent shifts with shared layout guards.
Original radius/padding/phase budgets and independent 1e-11 gates remain intact.
GPU1 stage16PASS/1FAIL31.148s and sole failed storage rerun1PASS8.702s are retained
separately; added zero-tolerance comparator was corrected after independent
review, exact metadata checks and same-instance observations, without production
changes or repeating sixteen passes. A pre-run axis-lifetime P2 was closed by
restoring default expression and releasing strict intermediates; actual lifetimes
and nine necessary separable regressions passed. Owned p12→3/L9 is60,928B versus
291,328B dense, shared edge7,168B; not universally smaller or bit-exact. Reference
and assessment: docs/{reference,assessments}/sie_fmm_compact_changing_carrier.*.
Next actual variable-order support-tree sharing/preflight, then cones/global
admission/public/periodic/resources/Nscaling/physics/populations/fullfourrepo/
>=2FINALaudits. This is not production qualification; GSP-01 and original failures,
final scope and thresholds remain OPEN/unchanged. Goal ACTIVE.

## 2026-10-02 — GSP-09 compact variable-carrier support tree

Explicit compact_phase_v1 tree caches phase/center layouts once per order key;
edge excludes shared layouts, tree counts them. Full capacity preflight and actual
support/padding/phase/current/H contracts preserved. GPU1 19PASS410.813s/rc0/
405stable,7new+12necessaryoriginal tree regressions; no unrelated repeats. Four
map ASTs/395otherprod/originaloracle sources unchanged. Highordertrees each1440784B
vs identicalhistoric49960112B; selected weak/originalH<=2.882656274965703e-11<1e-8,
independent allnode/current/higher<=5.0584164037800926e-15<1e-11. Exact unique
cache inventory/budget/mutation/savedbackward checks pass. Assessment/reference
sie_fmm_compact_changing_carrier_tree.* bind receipts and historical comparisonSHA.
Next continuous kernel/physical-gradient directional budgets and weighted phase
path/global/native/quadrature errors before cones/public/periodic integration.
Whole-system resources/crossover/scaling/physics/populations/final4repo/≥2FINAL
remain OPEN; GSP-01 failures, Goal full scope and all gates unchanged. Goal ACTIVE.

## 2026-10-02 — Continuous ideal directional kernel admission

Primary fmm_error continuous bivariate Cauchy Green and physical-gradient PERCOMPONENT
bounds support optional directional FFT scalar/all-row batch admission before
coefficients, exact16B independent radii and semantic/savedbackward guards. Bound
workspace includes17validfixed/adaptivecontours; availablepositive tails neverzero.
GPU1 installedCore stage21PASS/1FAIL26.904s/405stable preserved: axialp6 ideal
gradbound1.0066621761278174e-8>unchanged1e-8; noactualerror-exceeds-budgetproof.
Onlyfailedp6 rejection+new samecasep7 ran:2PASS24.978s/399stable; p7gradbound
8.388851575035643e-10, actualfourfieldrelative<=1.7707086541095904e-11.
21priorPASS notrerun; productionunchanged. Loworder/EMcomputedbound*1.01 are
NOTfixed1e-8EM/globalcontrols. Defaultcoefficient/FFT ASTs/errorprefix394otherprod
andoriginaloracle sources unchanged. Assessment/reference sie_directional_kernel_budget.*
bind47SHA and independentPHASEaudit, notFINALGoal. Supportsassumed, boundsideal
only: next weightedRWG/phasepath/global/native/quadrature budget beforecones;
public/periodic/resources/crossover/Nscaling/physics/populations/final4repo/≥2FINAL
remainrequired. GSP-01OPEN/allfailures/scope/thresholds unchanged; no productionqualification.

## 2026-10-02 — Historical original scalar D/K full resources

Frozen1c403d2/installedCore/GPU1 actual3840 fixedka scalar_dk_v1 collector
/tmp/gsp-scalar-dk-original-full-resource-v1 closed1PASS7621.401s/rc0/400stable,
zeroerrors/skips/failures. Original geometry/p11/policy/gates unchanged; original
admission/resourceSHA reused, no planning/defaultproducer repeat. Exactowned
1051291168/table354688320/near515446504/proj180911744B; actualnear+total reduction
469522944B andallothercategoriesunchanged. Owned>dense943718400:crossoverFAILED.
WholePROCESS setup/actions/backward peak allocated1224886272/reserved1317011456B
includeswarmups/difffwd/resetgaps; excludesdevice/solve/teardown andwholeSystemqual.
Adjoint1.6072017103952612e-15/current7.921092909612202e-17<1e-11; sharedGPUtiming
not speedup/Nscaling. Transferredtestedfixturebyteexact/no numericrepeat. Assessment
reports exactnewermainproductiondifferences: historicalreceipt, notcurrentqual.
Seeassessment/reference sie_fmm_scalar_dk_original_resources.* andindependent
PHASEaudit, notFINALGoal. Growing/FFT/directional/currentresources/crossover/
Nscaling/physics/populations/final4repo/≥2FINAL remainOPEN. GSP01OPEN/allfailures/
scope/thresholds unchanged; GoalACTIVE/no productionqualification.

## 2026-10-02 — Direct prescribed RWG weighted directional admission

Primary from_projections derives complete selectedplus/minus support radii and
reusescontinuouskernel/existingmaterial-balancedFrobenius formula beforecoefficients.
Explicitkernel1e-6/1e-4+idealRWGabsolute1e-4 controls are NOTfinalrelativegatechanges.
InstalledCore/GPU1 6PASS9.750s/rc0/398stable/zeroerrors/skips/failures:4new plusonly2
necessaryoriginaldefaultEMweak/H/current controls, no unrelatedrepeats.395otherprod/
originalprojectionAST/weakoraclebytesunchanged. Independentq16L2/peredgeaggregate
1e-11, complete5×3unsorted/shifted actualsupports; weak/H<=1.0232836701534997e-10
<1e-8, nativeH/current/dot/higher<=1.934975691901519e-15<1e-11. Idealweightedbounds
1.2556673124068132e-7/2.8336715573845234e-8; callerprojections excluded, exact
scalar13736B/16Bradii, fullbasisadmissionworkspace219648B. Preallocation/policy/
externalmap/geometry/budget/impedance/receipt/savedbackwardguards pass. Assessment/
reference sie_directional_rwg_budget.* bindexecution+source+diagnosticreceipts;
independentfinalPHASEauditnotFINALGoal. Momentoperator idealcompositiononly with
exactprescribedmaps, noFP/quadrature/phasepath/global/physical/productioncert. Next
phasepathaccumulation/native+quad/global allocation/cones/public/periodic; full
resources/crossover/Nscaling/physics/populations/final4repo/≥2FINAL remainOPEN.
GSP01OPEN/allfailures/scope/thresholds unchanged; GoalACTIVE.

## 2026-10-02 — Prescribed node ideal phase-path plus weighted admission

Native maxdescendant ∏(1+δedge)-1 distinguishesexactsentinels/positivefloors;
from_nested_nodes combinesphysicalG/eachgradient E+Δ(M+E), actualfullcorner radii
never shrink, materialbalancedselectednodestatistics anddoublebudgetsbeforecoefficients.
GPU1installedCore first4PASS124.234s/397stable/rc0 preservephysicalweak/H/current/
higher/guardcoverage. LateP2newimplicitabsolutepytestmathcomparison keptfrozenLIVE;
only2newstrictMP100 root/nodebranch/combinedformula checks2PASS8.777s/397stable.
Explicitabs0/rel1e-11, maxcombinedrel8.138041079178311e-16, pathmaxrel0; production
unchanged/all4coveredPASSnotrerun. Completecompactnonrootp8 300MHz6×6/q16vsq12
weak/H<=8.033058151751436e-14<1e-8; nativeH/current/higher/dot<=1.082470882250305e-15
<1e-11. Kernel1e-6/1e-4 andabsolute1e-4 arecontrolbudgets, NOTfinalgatechanges.
Balancedidealbounds7.068760208257873e-8/1.6377664972307507e-8; ownedscalar27552B/
16Bradii, no persistentpatharrays, externaltreesexcluded. Combined-not-purebudget
negative/preallocation/owner/transfer/savedbackwardcontrols pass.394otherprod/old
allfunctionASTs/wholenestedbyteprefix unchanged; no oldtree/FFT/resource/public
checksrepeat. Assessment/reference sie_directional_phase_path_budget.* bindinverse
strictcorrection/runtime/oracles/source; independentPHASEauditnotFINALGoal.
Idealexactprescribednodecomposition only; no finitequad/FP/global/physical/
resources/scaling/productioncert. Next globalfarpartition/cancellation-safe
allocation and leafquad/nativebudgets beforeactualcones/public/periodic/full
resources/crossover/Nscaling/physics/populations/final4repo/≥2FINAL. GSP01OPEN/
allfailures/scope/thresholds unchanged; GoalACTIVE.


## 2026-10-02 — Explicit disjoint directional FAR actions completed

The internal DirectionalNestedFarActions composes actual admitted complete-node
blocks across distinct carrier owners on shared basis/tree identities. Integer
half-open sweeps reject overlapping/duplicate physical rectangles without an N²
mask; CUDA log-RSS enforces a common-impedance ideal regional absolute budget.
Source-upward/node-M2L/receiver-downward and complete Hermitian/current/higher
maps cache each active owner once per physical RHS chunk and release generated
products. No newly owned retained CUDA arrays; caller dependencies are excluded.
Near coefficient slots remain zero: this is a FAR component, not a full system.

GPU1/installedCore evidence retains initial3PASS2FAIL93.85s (p6 kernel admission),
then only2FAIL12.66s (p8 default128MiB action workspace), then only2PASS652.33s
(p8 explicit512MiB workspace). Sessions68457/28854/16348 terminal and closed,
397 stable sources each, production unchanged across corrections; all original
failures/tests/log/XML/returncodes preserved and3coveredPASS controls not rerun.
Explicit kernel1e-6/1e-4 and ideal node/regional absolute1e-4 budgets, q12 maps/
q16 original references and final weak/H1e-8/nativecurrent1e-11 gates unchanged.
Two30-edge disjoint spheres/83nodes/3blocks/2opposite carrier owners at300MHz,
lossless+lossymagnetic: weak<=4.644375218209839e-14, originalH<=6.568807929616342e-14,
nativeforward/H/current/adjointcurrent/higher/dot<=1.3144393615093706e-15.
MP100 RSS comparisons useabs0/rel1e-11; ideal bounds4.776982970786096e-7 and
1.1416855496644979e-7. Independent766integercell cases, metadata/budget/caller
identity/externaltransfer savedbackward,15noncontiguousRHS/empty/lifetime and
injectednonfiniteoutput controls pass. Plannedcomponentworkspace147240864B,
owned0B; no wholeprocesspeak/crossoverqualification. Entireoldprimarybyteprefix/
23oldfunctionASTs/395otherproductionfiles unchanged; no already-covered tree/
phase/kernel/FFT/public/resource/Bench repeat. Evidence summary andfinalsource
scope /tmp/gsp-directional-far-actions-evidence-v1; assessment/reference and
FEATURE inventory accompany this stage. IndependentfinalPHASEaudit is not a
FINALGoal audit. GSP01OPEN/allfailures/scope/finalgates unchanged; GoalACTIVE.

Next coherent phase adds original packed near correction with exact integer
coverage and complete scalar exterior/body-local interiors using common-eta
B_ext+RSS(interior) rather than RSS across overlapping media. Reviewed design
/tmp/gsp-directional-complete-regional-proposal-v1.md. Leaf/native error budgets,
automatic cones/public/periodic/whole resources/crossover/Nscaling/physics/
populations/finalfourrepo/at least2FINALaudits remain required.


## 2026-10-02 — Complete prescribed directional near/FAR scalar systems

Original packed D/K near plus explicit disjoint directional FAR now proves full
coefficient pair coverage by half-open integer disjointness and total area before
integration. Actual scalar bodies bind complete exterior/interior regions and one
common eta; native ideal admission uses B_ext+RSS(disjoint interiors), not RSS of
overlapping media. Full H/current/higher actions, bounded caches, exact owned near
and embedding inventories and semantic/numeric saved-backward guards implemented.
GPU1 installedCore/nativeCUDA targeted4PASS18.469s/397stable/rc0/session90946 closed;
no covered FAR/tree/phase/kernel/FFT/public/resource/Bench repeat. Full24x24 lossy
magnetic two-body weak2.3857070725312304e-16/originalH2.9327541750632917e-16,
native/current maximum2.7836295418106014e-16, original1e-8/1e-11 gates unchanged.
All physical interiors are original near; actual positive interior FAR physics is
not qualified. Strict independent MP100abs0 overlap-norm control and all-FAR empty,
coverage/capacity/identity/mutation guards pass. Original near reference shares q4
quadrature and proves composition only. Owned near2776+1392+1392B/system192B;
planned system workspace91,119,680B excludes setup/callerI/O/cache/FFT, no total
resource/processpeak/crossover claim. Entire3primarybyteprefixes/23oldAST/393other
production files unchanged. Evidence and independent PHASE review paths:
/tmp/gsp-directional-complete-regional-evidence-v1/summary.json and
/tmp/gsp-directional-complete-regional-independent-phase-audit-v1.json.
Review is not FINAL Goal acceptance. GSP01OPEN/priorfailures/scope/gates unchanged.
Leaf/native certification, positive interior FAR controls, automatic cones/public/
periodic whole resources/crossover/Nscaling/physics/populations/finalfourrepo/atleast
2FINAL independent audits remain required; no production qualification granted.


## 2026-10-02 — Prescribed directional shared Riesz/Krylov integration

Actual complete DirectionalSystemActions/provider bind before native LocalRWGRiesz;
shared Krylov/current/residual/preconditioning/persistence reused, no rebuilding
or dense/H conversion. GPU1 installedCore/nativeCUDA targeted3PASS
20.27s/397stable/rc0/session38258 CLOSED, old primary whole prefix/2oldASTs/
395otherproduction unchanged; no covered physical/tree/kernel/public/resources
checks repeated. Two6-edge lossy magnetic bodies at400MHz, actualEx/Ey2RHS
GMRES/blockGMRES; full original dual<=1.4249006418883665e-12, dense current
relative<=1.5594300533200172e-11, native Riesz residual-current gradient
0.0 with unchanged1e-8/1e-7/1e-11 selected gates.
Both residual admissions pass, nofallback; one-step nonconvergence/directfallback
refusal/zero-current wrongresidual/preRiesz identity/savedactualaction+metric
guards pass. Raw allocator-since-external-reset fields are not wholeprocesspeak
proof, timings notscaling. Physicalinteriors allNEAR; originalq3near oracle shares
integration and qualifies composition only, not positive interiorFAR ornear physics.
Idealcommoneta bound notRiesz/residualcert; runtimeoriginalsample remainsnotglobal.
Evidence /tmp/gsp-directional-krylov-evidence-v1/summary.json; independent PHASE
review /tmp/gsp-directional-krylov-independent-phase-audit-v1.json, NOTFINALGoal.
GSP01OPEN/priorfailures/scope/gates unchanged. Leaf/nativebudgets, positive interior
FAR, automaticcones/public/periodic/wholeresources/crossover/Nscaling/physics/
populations/finalfourrepo/atleast2FINAL audits remainrequired. No productionqual.


## 2026-10-02 — Actual connected-body selected internal directional FAR controls

No production edits:396productionfiles byteequal. Actual connected480-RWG closed
sphere300MHzeps2mu1.3 lossless/lossy(.01sigmae/.02sigmam), commonactualexterior eta,
full960physicaldomain retained in prescribedcomplete6node supporttree. Twoordered
single-RWG internalFAR blocks, twooppositep12/q16 carriers and originalq20selected
4x4 reference, zeroomittednear/FARslots; notcompleteinteriorregion/systemphysics.
Initial2FAIL1PASS9.170s/397stable/rc1/session76645 CLOSED: actualweak
.8723484927515467/3.008941180965765 with length.01 remainunqualified/preserved.
Onlyfailed2PASS30.380s/397stable/rc0/session45280 CLOSED aftersingleexplicitlength
changes toactualD/(2p+2); allproduction/support/material/order/quadrature/budgets/
original1e-8weakH/1e-11nativegates unchanged. Kernel100/1e5 andideal.1 arenew
conservativegeometrycontrolbudgets, not finalgates. Inverse restoresoriginaltest
byteexact; coarsePASSfunctionASTunchanged/notrerun. Currentweak<=2.7822870141741786e-12,
originalH<=2.79032687459648e-12, nativecurrent/higher/H/dot<=3.632134563489498e-14.
15noncontigphysicalRHS/zerosavedinternals/exactzeroomittedoutputs/exactcaller
projectionbytes28,811,776+27,960,016 pass. NewFARowned0B/componentworkspace210,184,272B
excludescallerI/O/cache/FFTlibraries, no wholepeak/crossover/scaling claim.
Coarseactual120-RWG support ratio.633 unavailablep8/10/12 rejectszeroplaceholders;
480selectedratio.3047209342521525. Idealbound.05079/.04920 unchangedacrosslength,
notnativeerrorcert. Evidence /tmp/gsp-directional-interior-far-evidence-v1/summary.json;
independentPHASEaudit /tmp/gsp-directional-interior-far-independent-phase-audit-v1.json
isnotFINALGoal. No oldcoveredweak/tree/phase/kernel/FFT/public/resource/Bench
repeat. GSP01OPEN/alloldfailures/finalscope/gates unchanged. Actualcompleteinterior
near+nonzeroFARphysics/finiteleafnativebudgets/automaticnormalization/cones/order/
public/periodic/wholeresources/crossover/Nscaling/physics/populations/finalfourrepo/
atleast2FINAL audits remainrequired; no productionqualification.

### Bounded automatic directional cone/order planning completed

Primary fmm_order.py append only/old prefix and3 original ASTs unchanged/395other
production files identical. At most32 fixed CUDA float64 declared support pairs;
canonical dominant-axis cube keys, minimum passing p0..12 then coarsest grid;
available required before budget, all rows pass before selected primitive factory.
Automatic min(D/(2p+2), max positive support radius), zero-support distance fallback.
All flags remain false for physical/native/production/global qualification.
Original v1/session51749 CLOSED4FAIL2.92s/397stable (defaultfloat32sign). One dtype
correction v2/session99688 CLOSED2PASS2FAIL4.72s/397stable: independentMP100
selectionmax1.197930643570544e-13<1e-11/owned576B/strictcapacity+mutation PASS;
distance-only nativephysicalmax730073.4164/208.7741 FAILpreserved. Supportcap
v3/session53907 CLOSED onlyfailed2+newnormalization1 PASS3.89s/397stable, unchanged
physicaltestAST/kernel1e-8/relativephysical1e-8/native1e-11 gates. Actual p6/p5
grid32 fourfieldsmax8.330138812244727e-10, nativecurrent/higher0/dot4.00655e-16.
No priorPASS rerun. Plan144B/pair/primitive13736or9032B; componentworkspace excludes
caller/library/allocator, not wholepeak/crossover/scaling. Evidence
/tmp/gsp-directional-cone-order-evidence-v1/summary.json; independent PHASE audit
/tmp/gsp-directional-cone-order-independent-phase-audit-v1.json, notFINALGoal.
GSP01OPEN/alloriginalfailures/finalscope+gates unchanged. Full sparse tree contexts,
actualsupport/leafnative+quadrature/phase/global admission, complete interiorFAR
systemphysics/public/periodic/wholeresources/crossover/Nscaling/populations/current
fourrepo FINALacceptance and atleast2FINALaudits remainrequired; no qualification.

### Automatic actual-support directional RWG blocks completed

Primary fmm_directional_rwg/fft only;394 other production files unchanged. Two
original constructors share extracted exact structural capacity formulas with
new actual-support automatic direct block, inverse old module prefixes byteexact.
Complete selected plus/minus vertices and native U/Q material weights derive
conservative half-squared ideal budget/channel before automatic cone/order and
owned direct-map/PMCHWT factories. All ideal/capacity/quadrature checks before
integration/coefficients, actual support/geometry/medium/configuration guards.
v1/session25560 CLOSED7PASS10.21s/399stable (3new cases+4affected original only);
v2/session94699 CLOSED onlyNEW MP100 budget1PASS3.69s/397stable; all396prod and
originaltestprefixunchanged, old7PASS notrerun. Lossless/lossy automaticp6/p5->
tighteridealp10/p9, unsorted5x3 actualRWGs/common exterior eta376.7303136668617.
Originalweakmax4.032152582963799e-9/fullH5.640352022062615e-9<1e-8, nativecurrent/
higher0/dot2.4344881870058463e-15<1e-11/15noncontigRHS/no savedM2L internals.
IndependentMP100 effectivebudgetsmaxrel4.489116308638273e-15/combined2.35306590941802e-15
<1e-11; unsplit sqrt2counterexample verified. Owned100096/66720B includes184plan+
generatedinputs, geometryexcluded; workspace119389776/85897296B componentonly,
notwholepeak/crossover/scaling. Evidence /tmp/gsp-directional-actual-support-evidence-v1/summary.json;
independentPHASEaudit /tmp/gsp-directional-actual-support-independent-phase-audit-v1.json, notFINALGoal.
GSP01/GSP09OPEN/GoalACTIVE/alloldfailures/finalscope+gates unchanged. Actualcomplete
interiornear+FAR/sparsecone-treecontexts/finitequadrature+native/phase-global/public/
periodic/wholeresources/crossover/Nscaling/physics/populations/finalfourrepo/atleast2
FINALaudits remainrequired; no productionqualification.

Initial PHASEaudit2P2 preserved: unbounded wrapper intermediates and outer
saved-backward guard bypass. Unified correction changes only new class3action
methods: reuse4RHS chunks +shared fixed linear protocol. v3/session4370 CLOSED
onlyNEW changedpath3PASS8.88s/397stable; old7PASS+MP1PASS notrerun/originalweakmatrix
SHA reusedinplace/no integrationrepeat. Hmax4.965299572218501e-9/native1.8707764225587183e-15,
alloriginalgatesunchanged; wholewrappersaved0/maxchunk4/lifetime+empty+14saved
backwardtripwires PASS. Final consolidated evidence
/tmp/gsp-directional-actual-support-evidence-v2/summary.json; correctedPHASEaudit
/tmp/gsp-directional-actual-support-independent-phase-audit-v2.json; notFINALGoal.

Finaltestportabilitycorrection removes unconditional historical/tmp dependency:
optional explicitSHA-bound reuse, defaultfreshsameq16originalreference/no skips.
OnlyNEW lifecycle v4/session56849 CLOSED1PASS2.42s/397stable/all396prod unchanged;
no covered numericalrepeat. Final evidence
/tmp/gsp-directional-actual-support-evidence-v3/summary.json andPHASEaudit
/tmp/gsp-directional-actual-support-independent-phase-audit-v3.json.

### Prescribed sparse directional RWG context forests completed

Primary fmm_directional_moments only/395 other prod unchanged. Reverse3context
cardinality expressions restoresentireoldprefixbyteexact; old fullowners counts
equalphysicalnodecount. New selected(node,carrier/order) descendantclosures only,
shared samefamily ancestor/descendantcontexts, uniqueorderlayouts, actualsupport
andglobal mixedleafroutes; no fulltreepercone. Allfourmap/current/higher methods
are shared primary methodobjects, no duplicate numeric kernels. Sparseindices
explicitlynotoldfulltreetype; oldEMconstructorrejectsearlyuntilcorrectsparsebinding.
v1/session97374 CLOSED5PASS42.61s/398stable (4newcases+1necessaryaffectedoldsingle);
v2/session22256 onlyNEWconfig1PASS3.45s/397stable, oldprefix inverse/testprefixexact;
v3/session52254 onlyNEWfixedmodes1PASS6.53s/397stable;v4/session55099 onlyNEWprotocol
1PASS3.45s/397stable; no earliercoveredPASS repeats. Independentq16 sparsefourmap
max1.907498768330214e-15<1e-11/current+higher0/no savedwrapper/15noncontigRHS/missing
cotangents/empty/20omittedphysicaledgesexactzero/capacity+savedmutationmodeguards
PASS. Active12/11vsfull-treepercarrier78/102slots; sameorderlayoutsharedonce.
Owned72224/84216/423696B/componentworkspace4031376/4151376/4534752B exclcaller/library/
allocator/notwholepeak/crossover/scaling. Mathematical samefamilyphase truncation0
notnativecertification. Evidence /tmp/gsp-directional-sparse-contexts-evidence-v1/summary.json;
independentPHASEaudit /tmp/gsp-directional-sparse-contexts-independent-phase-audit-v1.json,
notFINALGoal. GSP01/GSP09OPEN/GoalACTIVE/alloldfailures/finalscope+gates unchanged.
Next actualsparsecoupling/kernel+weighted+phase-global/automaticpartition-order
contexts/completeinteriornear+FAR/nativequad/publicperiodic/wholeresources/crossover/
Nscaling/physics/populations/finalfourrepo/atleast2FINALaudits allremainrequired.
No production/physics/complexity qualification.

### Sparse actual-medium directional M2L and FAR completed

Primary fmm_directional_rwg only/395 other production files unchanged; shared
support-radius/complete FAR initialization inverse recovers old module byte equal.
Sparse material-bound factory uses explicit active context IDs and separately
records physical nodes; actual complete supports, common exterior eta, continuous
G/per-gradient and weighted U/Q ideal admission before coefficients. Fixed family
ideal phase truncation0 excludes native phase/shift/quadrature. Shared old FAR
numerical/current/adjoint/higher protocol, physical disjoint rectangle sweep and
RSS aggregation; no duplicate PMCHWT kernels or full-tree-per-cone maps.
v1/session72117 CLOSED6PASS129.832s/401stable/rc0:4new+2necessary affected extraction
checks. v2/session85665 onlyNEW unequal-order/native-mode/saved-adjoint1PASS12.693s/
398stable/rc0; sparse-only guard inverse/test prefix exact, original6PASS not rerun.
Actual30+30RWG spheres300MHz eps2/mu1.3 lossless/lossy sigma_e.01/sigma_m.02 p8/q12
vs originalq16: weak<=3.63859e-14/H<=5.52113e-14<unchanged1e-8, native/current/higher/
dot<=1.11253e-15<1e-11/no savedwrapper/15noncontig/omitted edges exact0. MP100
independent triangle/material weighted relative<=2.72581e-15<1e-11 and RSS abs0.
Physicaltree83/active24 vs hypothetical166; owned map640640B/couplings82656B/new FAR0B,
component mapworkspace19985168B/FAR103484288B excludes caller/library/allocator,
not wholepeak/crossover/scaling. All preflight/saved configuration/mode guards,
physical4RHS/cache lifetimes/empty backward PASS. Three original GPU jobs still LIVE.
Evidence /tmp/gsp-directional-sparse-coupling-evidence-v1/summary.json; independent
PHASE audit required before commit, not FINAL Goal audit. GSP01/GSP09 OPEN/Goal
ACTIVE/all original failures/final scope+gates unchanged. Next automatic sparse
partition/cone/order and full regional/system integration; complete interior near+
FAR/nativequad/global/publicperiodic/whole resources/crossover/Nscaling/physics/
populations/final fourrepo/at least2 FINAL independent audits remain required.
No production/physics/complexity qualification.

### Complete sparse directional near/FAR system and Riesz/Krylov completed

Primary fmm_regional only/395 other production unchanged/whole module inverse
exact/five original action/state/bound ASTs unchanged. Accept admitted sparse FAR;
near IDs always complete physical tree, including nodes outside active context ID
range. Shared packed original D/K/full coverage/actual body+medium/common eta/
regional+system/native guarded Riesz/Krylov; no duplicate physics implementation.
Same closed physical slender boxes: coarse18/body near_tol1e-5/max16 fails.0193,
v1/session30936 CLOSED old affected1PASS+4ERROR7.30s/399stable, oldPASS not rerun.
Refine same geometry longitudinally to198/body, max32/tolunchanged, all396prod and
four original gate ASTs unchanged. v2/session10796 CLOSED3PASS1FAIL541.59s/398stable:
whole originalq6/singular3 weak2.77639e-16/six independentq16 FAR≤1.94774e-13,
originalH6.90310e-16/native-current-higher≤1.70104e-15/no savedwrapper/15noncontig,
MP100 ideal triangle+RSS abs0, physical coverage/capacity/saved-system+Riesz PASS.
Both actual eps2/mu1.3 and eps3/mu1.5 lossy interiors have2 positive FAR+7near;
exterior2crossbody FAR+34near, complete396 edges/792 physical currents preserved.
Diagonal-only GMRES original dual.275741/current.956607 FAIL retained. Existing
bounded original32-edge inverse: v3/session52653 CLOSED NEW reference lifecycle
1PASS+solve1FAIL442.429s/400stable, full original2.33421e-10/current2.10933e-9 but
192 steps do not meet inner1e-12. v4/session27812 onlyFAILED solve capacity768/384,
all original tolerances/material/geometries/q/p/preconditioner/reference unchanged.
No previousPASS repeat/oracle reintegration: optional identity+SHA-bound original
matrix1940060e97fb5df1341c3d87b51f5577e0ee670883b839d15cfe258020f272d5 reusedinplace,
portable fresh ordinary default/strictwrong-SHA/identity/missing/partial lifecycle.
Owned near5094936/1273632/1273632B/maps84944each/couplings55104each/indices6336B,
not wholepeak/crossover/scaling. All original failures retained/unqualified.
Evidence /tmp/gsp-directional-sparse-system-evidence-v1/summary.json; independent
PHASE audit /tmp/gsp-directional-sparse-system-independent-phase-audit-v1.json,
not FINAL Goal. GSP01/GSP09 OPEN/Goal ACTIVE/original final scope+gates unchanged.
Automatic sparse partition/cone/order/changing-carrier DAG/global certification,
complete nativequad/physical/publicperiodic/whole resources/crossover/Nscaling/
populations/final fourrepo/at least2 FINAL audits remain required. No qualification.

v4 CLOSED1PASS/511.75s pytest/512.9551367191598s runner/400stable/rc0/no failures/errors/skips. All phase sessions drained and actual processes absent; original three GPU jobs remain LIVE.

Configured solve results: [{"method": "gmres", "original_full_dual": 7.778820926838563e-13, "current": 3.2365996804172426e-12, "iterations": 367}, {"method": "block_gmres", "original_full_dual": 1.4353432449029596e-12, "current": 5.86589436237549e-12, "iterations": 127}]

### Automatic complete sparse directional regional planning completed

Primary fmm_order appended auto actual support tree/complete shared sphere partition,
material/RWG weighted pair B/sqrt(F), half-square G/grad budget, shared minimum-order/
coarsest-cube selection/canonical carrier+order/active requests/native final RSS.
Build uses shared sparse maps/actual medium FFT M2L/original D/K near; no duplicate
physics kernels. Four primary sparse owner empty-forest constructor expressions;
old module inverse byte equal/old order prefix byte equal/394 other prod unchanged.
All-near owners allocate0B/no maps/layouts/shifts; complete physical domain retained.
v1/session27340 CLOSED5PASS8.91s/401stable/rc0:4new+1necessary affected nonempty guard.
Two actual lossy magnetic6-edge tetra bodies300MHz, complete12edges/exterior2FAR+
2near/two all-near interiors, automatic p7/two families. Originalq4/singular3 whole
weak3.07380e-16/two independentq16 FAR<=7.63734e-13<1e-8/H2.87939e-16/native current+
higher0/dot2.93129e-16<1e-11/no savedwrapper/15noncontig. MP100 independent actual
material limits/exhaustive cube/order/RSS max7.10462e-16<1e-11/abs0. Complete physical
coverage/allnear empty/no factory capacities/order/quadrature/family/guards PASS.
Initial independentP2 localmedium physical value binding closed by state-only
frequency/eps/mu/k/eta +treeversion/leaf/morton. v2/session77918 onlyNEW guard1PASS
3.37s/400stable/rc0; reversefix recovers entiretestedv1/order, oldtestprefix exact,
old5PASS not rerun. Both sessions drained/actual processes absent; original3LIVE.
Exact plan inventories1016/216/216B include owned tree/catalog; separate component
workspace excludes caller geometry/Python metadata/library/allocator/solver/oracle,
notwholepeak/crossover/scaling. Evidence /tmp/gsp-directional-automatic-region-evidence-v1/summary.json.
IndependentPHASEaudit required beforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN;
alloldfailures/finalscope+gates unchanged. Automatic positiveinteriorFAR/global
native+quadrature/changingcarrierDAG/publicperiodic/wholeresources/crossover/Nscaling/
physics/populations/finalfourrepo/atleast2FINALaudits required. No qualification.

A final NEW planner-only control stresses actual material tightening rather than explicit kernel clipping. Starting from loose G/grad budgets1/1 and ideal regional B=1e-10, both ordered FAR pairs receive independent100-digit derived limits G=1.63813114568e-10 and grad=1.01635123603e-6. Exhaustive cube/order selection still gives p7; material-weighted pair limits and final RSS relative errors are at most4.26696396932e-15 with absolute tolerance0. The map factory is forbidden throughout. v3/session78499 CLOSED1PASS3.69s/400stable/rc0, all396 production files byte equal to tested v2, entire v2 test prefix SHA exact. No existing6PASS, maps, near physics, original matrix or FAR oracle is repeated. The independent final phase audit includes this supplemental evidence.

### Prescribed shared changing-carrier RWG DAG completed

Primary fmm_directional_moments only/395 other prod unchanged. Sharedinitializer
DAG=False exactly original sparseconstructor AST; original two actionASTs exact
when _transfer lookup restored;23otheroldmethodAST exact. All8native/currentaction
methodobjects shared. Explicit canonicalreachabletopological physical contexts/
complete direct child coverage/sourceorder>=parent+L/phase<=12; multi-parent shared
child integratedonce/sourceupwardonce/downward sumsallparent locals beforedescend.
Existing compactchanging transfers keyed(parent,child), uniquephase+centerlayouts;
actualsupport+phase+capacity beforemaps. DistinctDagtype oldphase0M2L refuses.
v1/session39713 CLOSED5PASS1FAIL178.832s/401stable/rc1:3newactual mapcontrols+2necessary
oldprotocolPASS; negativeM2L omittedkeywordsFAIL preserved. v2/session23034 onlyFAILED
1PASS4.959s/399stable/rc0, all396prod unchanged/oldpassinggate+fixtureAST exact; old5PASS
notrerun. Actual30edge sphere/global60/mixedfullleaf underdeclaredk5+.8j maps only:
allfourindependentq16 maps+H max3.37472e-15/nativecurrent5.23087e-17/higher5.52108e-17
<1e-11/15noncontig/no savedwrapper/missingcotangents/sharedlocals sum. MP100 allcontext
maxphaseproducts exact0orpositive rel<=4.68365e-15<1e-11/abs0/notindegree multiplying.
Owned1244568/2059560/1723672B/context40vs78,52vs102,3vs4; componentworkspaces29715584/
32858544/26215152B excludescaller tree/geometry/metadata/wholeI/O/allocator/library/
solver/oracle, notwholepeak/crossover/Nscaling. Additionalimportedoldhelper absent
frominitialfrozeninventory honestlyseparateimmutablebaselineblobproof/notrewritten.
Bothcollectorsdrained/actualprocessesabsent. IndependentPHASEaudit beforecommit/
notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/finalscope+gates unchanged.
Next phase-inclusive actualmaterial DAGM2L/fullnear+FAR/automaticchildconesorders/
fullnativequad/publicperiodic/wholeresources/crossover/Nscaling/physics/populations/
finalfourrepo/atleast2FINALaudits. No production/materialweak/complexity qualification.

### Material-bound changing-carrier DAG FAR coupling completed

Primary fmm_directional_rwg shares the existing phase majorant and sparse factory;
395 other production modules byte equal and37 original method ASTs exact. Distinct
DAG material admission includes both maximum path products before effective
kernel/RWG material budgets; physical support/context IDs separate, phase workspace
before statistics and effective budget failure before coefficient allocation.
Disjoint DAG FAR owner shares original physical/full-H/current/higher actions.
Actual300MHz eps2/mu1.3 lossless/lossy .01/.02 sigma, two30-edge spheres, common
exterior eta. Two selected6x4/4x6 physical rectangles in120 current coordinates;
not complete near/FAR or global operator certification. p8->12/L4 with signed
neighboring carriers/8requests/28contexts/20sharedchildren/40transfers/tree83.
Originalq16 weak<=3.70824e-14/fullH<=5.15265e-14<1e-8, native<=6.01163e-16<1e-11,
15noncontiguous RHS/no saved intermediates. MP100 phase/material/RSS<=5.16627e-15
relative1e-11/abs0; phase defects6.1e-26/6.8e-26, low-order measurable positive
phase-only budget rejection before coefficients, capacity/protocol/saved-F/H guards.
v1/session62970 CLOSED2PASS3FAIL21.696s/402stable/rc1 retained. Only failed fixture
signed carrier routing and invalid-ID call corrected; all396prod/physical gates
unchanged. v2/session76332 only3FAIL CLOSED3PASS163.631s/399stable/rc0; old2PASS
not repeated. Both sessions drained/processes absent/all transitive imports frozen.
Component map1478488B/couplings55104B/action0B; map/action workspace59163192/
155892360B excludes caller/allocator/library/solver/oracle, notwholepeak/crossover/
Nscaling. Independent PHASE audit required beforecommit/not FINAL Goal audit.
Evidence /tmp/gsp-directional-dag-coupling-evidence-v1/summary.json. GoalACTIVE,
GSP01/GSP09OPEN; all original failures/final scope/gates unchanged. Next complete
DAG near/FAR/automatic child cones/orders/nativequad/publicperiodic/whole resources/
crossover/Nscaling/physics/populations/finalfourrepo/atleast2FINALaudits required.
No production qualification. Earlier automatic magnetic actual collector remains
frozen in its original provider; its terminal acceptance is still pending.

### Automatic complete magnetic sparse interior controls completed

Tests/evidence only; no production change in this stage. Genuine numerical baseline
21e46ff retained in immutable collectors; after terminal source reconciles to
auditedf747466 by394byteequal modules and exact legacy constructor/lookup plus
inlined DAG=False sparse factory/admission/workspace/sparse=True FAR initializer
AST proofs;37 original class methods exact. No long run repeated/no current full
regression or production qualification. Original reference identity/SHA-bound
matrix1940060e97fb5df1341c3d87b51f5577e0ee670883b839d15cfe258020f272d5 reused
inplace/no reintegration/duplicate archive; all transitive helpers frozen.
Two198-edge closed magnetic boxes300MHz, eps2mu1.3sigma.01/.02 and eps3mu1.5sigma
.015/.005; originalgeometry/q6singular3nearmax32/commonexterior eta/gates unchanged.
Automatic leaf8/ratio.1/cones(2):exterior550FAR5124near/26families p8..12, both
interiors62FAR1469near/8families p7..10; every physical pair covered exactly once.
Plan/session21331 CLOSED1PASS32.283s/399stable/rc0. Actions/session95502 onlyNEW
CLOSED1PASS5029.069s/401stable/rc0; oldplanPASS not repeated/both handles drained/
actualPIDs absent.15noncontig complete-domain original forward1.29362e-15/H5.18661e-16
<1e-8/native current7.58317e-17/higher8.40392e-17/dot1.99502e-16<1e-11/saved0.
Sampled actions not a full-matrix norm certificate. Six first/last selectedq16
FAR probes max6.78076e-12<1e-8; remaining FAR quadrature uncertified.
Plan owned131032/26288/26288B/near4911752/1428176/1428176B/regional component
workspace1323360784/245560688/250833520B/explicit2GiB; excludes caller/allocator/
library/solver/oracle, notwholepeak/crossover/Nscaling or qualified performance.
Evidence /tmp/gsp-directional-automatic-interior-evidence-v1/summary.json; PHASE
audit beforecommit/not FINAL Goal. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/final
scope+gates unchanged. Automatic shared changing child cones/orders/fullnativequad/
publicperiodic/wholeresources/crossover/Nscaling/physics/populations/finalfourrepo/
atleast2FINALaudits required. No production qualification.

### Complete prescribed material DAG system completed

Only originalregional import/typegate admits distinctDirectionalDAGFarActions;
entireprimaryinverseexact/12oldAST/395otherprodunchanged. Sharednear/system/native/
Riesz/Krylov kernels untouched. FourNEW controls CLOSED4PASS/rc0/XML1383.867s/
runner1385.1786307068542s/403stable inclfivehelpers+explicitrefmatrixmanifest;
session30076drained/2203044+2203200actualpsabsent. Genuine numericalbaselinef747466
retained; terminal-onlyrebase db60224e addsoldautomatictest/docs only/all396prod
interveningexact/no numericreplay/no currentfullregression. Originalmatrix1940060e
97fb5df1341c3d87b51f5577e0ee670883b839d15cfe258020f272d5 reusedinplace/no duplicate.
Two198-edge closed magneticboxes300MHz/originalq6sing3nearmax32/commonexterioreta.
Prescribedcomplete-support childtrees; eachregion12ctx/4shared/8transfers/4maps,
rootorders8/child12/L4positivechangingcarrierpaths; sixactualFAR/near34+7+7/full
physicalpaircoverageonce. Whole792-column originalweak2.77639e-16<1e-8/sixq16FAR
max1.95395e-13<1e-8; originalnearquadratureshared/compositionnotphysicalconvergence.
15noncontig fullcurrents originalH6.92767e-16/nativecurrent5.04269e-18/higher
1.22340e-16/dot2.05765e-15<1e-11/saved0. StrictMP100phase max6.33948e-15rel1e-11
abs0/systemexterior+RSS9.28022e-14 idealonly/nativequad/globalnotcertified.
GMRES+blockGMRES actualEx/Ey/fulloriginaldual+current gatesunchanged;
rtol1e-12/atol0/max768/restart384/boundedleaf32originalfactors/DAGactions no fallback.
Oneiterfailure/directfallbackrejection/physicalcoverage+capacities+actualmedium/
commoneta/saved8fullH+DAGphasemode+Rieszmutation guards PASS. Near5094936/1273632/
1273632B/maps505048each/couplings55104each/embeddings6336B/systemcomponentworkspace
152545224B excludingnearsetup/caller/allocator/FFTlibrary/notwholepeak/saving/
crossover/Nscaling. Evidence /tmp/gsp-directional-dag-system-evidence-v1/summary.json;
independentPHASE audit beforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/allold
failures/scope/gates unchanged. Automaticsharedchangingchildcones/orders/global
nativequad/publicperiodic/wholeresources/crossover/Nscaling/physics/populations/
finalfourrepo/atleast2independentFINALaudits remainrequired. No production qualification.

Actual solver results: [{"method": "gmres", "original_dual_relative": 7.783996510071385e-13, "dense_current_relative": 3.237148175109244e-12, "iterations": 367, "fallback_used": false, "original_global_residual_certified": false}, {"method": "block_gmres", "original_dual_relative": 1.4356491055488424e-12, "dense_current_relative": 5.8676952349760104e-12, "iterations": 127, "fallback_used": false, "original_global_residual_certified": false}]

### Automatic shared directional DAG selected FAR completed

Primaryorder appendplanner; primarymoments shares exactoriginalcapacityformula,
SparseinitializerinverseASTexact/oldordermoduleprefixbyteequal/394otherprodexact.
Finitecandidate/contextscratchbounded; fixedphysicalnear/FAR+rootcones/orders;
allancestorpathproducts+allrootE+delta(M+E)/kernel/materialperpair/regionalRSS
recomputed beforemaps/coefficients; completeclosureleaf/bodymaps/uniquephasecenter
layouts/transfers/workspace scored; deterministic no rejectedFAR->near fallback.
Planningv1 CLOSED4PASS/399stable/2NEW+2necessaryoldcapacitycontrols; importedlegacy
oraclehelpernotinoriginal399inventory/honestbaseline+terminalexactnotretrofrozen.
PreliminaryP2plannerbindingpreserved/fixedonlyafterterminal; optionalboundguard
identity/config/current chain no actionkernelcopies. Actualv2 CLOSED3PASS/403stable
2NEW+1necessaryolddefaultguard. Boundaryv3 CLOSED2PASS/400stable/onlyNEW/all396prod
unchanged/actualtestexactprefix. 9distinctPASS/coveredPASS not repeated/no fullreg.
Actual3closed30-edge spheres300MHz/middley1e-6/vacuumEXTERIOR; distinctmagneticbodies
do not establishpositiveinteriorDAG. Auto65node/10FAR507near/6rootfamiliesp7+p5/
4positivechanginglinks/4sharedchildren/178ctx/166transfers/104leafcontexts;
allphysicalpairsonce/strictMP100allancestors+allrootphasekernelmaterialRSSrel1e-11
abs0max1.91695e-15. Originalq16all10selectedFAR complete180coordFAR-onlyweak4.40775e-11
<1e-8/H6.11756e-11<1e-8/nativecurrent0/higher4.65036e-16/dot5.02826e-15<1e-11/saved0.
Nearmatrixzero/NOTcompletefullMaxwellsystem/no duplicateweakarchive. Boundplanner
links/phasebudget/rootpolicy/removedguard rejectsreceipt/fwd/H/savedH. Exactmixed
p7+p5case4shared/zeroactualphase; paddedp12infeasiblepositivecasecorrectexactbaseline/
no savingclaim; finite1candidatepreservesphysicalpartition/reportslimit.
Wholeclosurecomponentcost39663504->35030768B/map4794192->3918960B/workspace34869312
->31111808B/candidatescratch1155584B/exclusionsrecorded/notwholepeak/perf/crossover/
Nscaling. Prototypeorder12/catalog32 are notGoalnarrowing. Evidence
/tmp/gsp-directional-automatic-dag-evidence-v1/summary.json/PHASEauditbeforecommit
notFINALGoal. Independentcovariancev4 CLOSED1FAIL72208.719s/715stable/handle98167
drained/actual846722+846726absent/DyadicConvergenceErrorbeforecovariancegate/no
restart/identitynotcollectednotinvented. Originaltensorcell+publictensorACLIVE
unchanged. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/scope/gatespreserved. Complete
automagneticnear+FAR/solves/nativequadglobal/publicperiodic/tensornonlinear/
wholeresources/crossover/Nscaling/physics/populations/currentfourrepo/atleast2FINAL
independentaudits remainrequired. No production qualification.

Terminal collectors: [{"path": "/tmp/gsp-directional-automatic-dag-planning-v1", "tests": 4, "XML_seconds": 78.128, "runner_seconds": 79.61735257599503, "source_count": 399, "status": "CLOSED", "returncode": 0}, {"path": "/tmp/gsp-directional-automatic-dag-actual-v2", "tests": 3, "XML_seconds": 408.961, "runner_seconds": 410.199377710931, "source_count": 403, "status": "CLOSED", "returncode": 0}, {"path": "/tmp/gsp-directional-automatic-dag-boundaries-v3", "tests": 2, "XML_seconds": 8.151, "runner_seconds": 9.263014268130064, "source_count": 400, "status": "CLOSED", "returncode": 0}]

### Complete automatic directional DAG discrete system completed

Test/evidenceonly/all396prodexact76927fc/401frozen completehelperclosurestable.
Session75825 CLOSED1NEWPASS/XML585.093s/runner586.4474143227562s/rc0/drained/
2294416+2294639actualpsabsent; coveredFAR/PASS/fullreg not repeated.
Three30-edge radius.02 closed spheres300MHz/x0,1,2middley1e-6/eps2,3,4mu1.3,1.4,
1.5sig_e.01sig_m.02/vacuum. Originalnearq6sing3pair4far16max32/tolerance/commoneta
unchanged; auto leaf4ratio.15cube2^20/exterior10FAR507near/4positivechanginglinks/
4sharedcontexts; eachactualmaginterior169nearZERO FAR/notpositiveinteriorevidence.
Full180-column originalweak assembledonce16edgetiles/no existingfulloracle/no
matrixarchive; weak2.24047e-15/H2.51689e-15<1e-8/nativeH4.98176e-16/current7.41389e-17/
higher1.59685e-16/Riesz9.38182e-17<1e-11/15noncontig/saved0. Wholeoriginalnearquad
shared/discretecompositionnotconvergedphysics. Planlinks savedcompleteH+receipt
guards PASS; solveacceptanceNOTRUN. Idealexterior+RSS6.31969e-6 excludesfinitequad/
native/globalcert;embedding1440B/systemcomponentworkspace85952048B/exclusions
recorded/notwholepeak/perf/crossover/Nscaling. Evidence
/tmp/gsp-directional-automatic-dag-system-evidence-v1/summary.json/independentPHASE
beforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/allfailures/scope/gates
unchanged. PositiveautoMagInteriorDAG/solves/publicperiodic/tensornonlinear/
nativequadglobal/physics/wholeresources/crossover/Nscaling/populations/current
fourrepo/atleast2FINALindependentaudits remainrequired. No production qualification.

### Guarded automatic DAG path admission reuse completed

Primarymoments2newbranches: exactnumericalkeys/supportbeforemaps plus borrowed
immutableplannerCUDAallancestorarray afteroriginalworkspace/currentguard. Default
prescribed traversal/maps/transfers/FFT/actionkernels unchanged; inversewholeprimary
byteexact76927fc/395otherprodexact. Bothcollectors genuinelytested76927fc; intervening
533f5928tests/docs only/all396prodbyteexact; originalsourcehashesretained.
v1CLOSED2NEWPASS/XML38.473s/runner39.709867808036506s/401stable; emptyv2CLOSED1NEWPASS/
XML3.566s/401stable/onlyNEWtestappend/exactv1testSHAsnapshot; handles77795+69516drained/
actualPIDsabsent/installedCore/GPU1. No oldPASS/q16/fullregreplay.
Actual3sphere90-edge exterior borrowedvsuncachedrel1e-11abs0/MP100allancestorsmax
6.64201e-16/exactzero0/32requestsidenticalarray/no2ndcache. Pathmode/identity/links
receipt/query/savedsourcegradient guards andpathversion savedreceiver guard PASS.
Actual30-edge magneticinterior ZERO FAR/emptyCUDAidentity/0ownedmapbytes/workspace
andmodeguards PASS/notpositiveinteriorFAR. torch.equal exactnumericalnotbyteidentity.
Map3918960B/componentworkspace31111808B unchanged; plannerownedstorageexternal
dependencycountsaggregate/notwholepeak/perf/crossover/Nscaling qualification.
Evidence /tmp/gsp-directional-admission-path-cache-evidence-v1/summary.json;
independentPHASE auditbeforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/allold
failures/scope/gatesunchanged. PositiveautoMagInteriorDAG/solves/publicperiodic/
tensornonlinear/nativequadglobal/physics/wholeresources/crossover/Nscaling/
populations/currentfourrepo/atleast2FINALindependentaudits remainrequired.
No production qualification.

### Public automatic scalar directional DAG integration completed

DirectionalFMMConfig strictnewversion/allfieldsbind; existingCartesianpayloads
unchanged. Primaryautomaticglobalfactory G/2 exterior+G/(2sqrt(bodycount)) eachactual
interior, exterior+RSS checkedbeforemaps/near and embeddingcapacitybeforeplanning;
primarynative map/FFT/actions/Riesz/Krylov reused. PublicScene->Simulation->Result
free-space scalarcomplex128/iterative nofallback only; periodic/tensor/HB remain
requiredOPEN. Auto/prescribedmetadata typedowners safe.390otherprodbyteexact5efbc2d/
inverseexistingprimarypaths and ASTproof/threecollectors396prodidentical.
FourcollectorsCLOSEDrc0/stable398+401+397+473/10distinctPASS=7NEW+3necessaryaffectedold;
coveredq16/native/fullregnotreplayed. QuickpytestPIDsnotobserved/notinvented;
originalhandlesdrained/knownrunnerabsent/childwaitedrc0. Publicactual2355117+2355251
absent/36704drained. Public3sphere90edge/180currents300MHz/eps2,3,4mu1.3,1.4,1.5/
sig_e.01sig_m.02/q6sing3pair4far16nearmax32tol1e-5; exterior10FAR/4positivechanging
links; threeallnear magneticinteriors ZERO FAR/notpositiveinteriorqualification.
Originalbounded16leaf GMRES115iterations/rtol1e-12atol0max240restart120/nofallback;
compresseddual1.11552e-12/original8edgesample1.57532e-12<1e-8/notglobalresidual/current
error/physicscertificate. ExactResultcurrents/configrestore withconstructors/solve
forbidden; policytamperrejected by numerical/operator/RHSidentity/schema11.
Bench69produnchanged; independentphysical/configdeclaration + nativeidentity,
exactJ/Mbytesarchive/reusedproducerSHAee789a6f2b369fc1fb108dc0ef28859436871a8e5c4ec72de8b670e5bd9c1c62/
noadditionalassemblyor solve/unqualifiedcandidateonly. Regionalcomponentcapacities/
plannerexternalstorage/notwholepeak/perf/crossover/Nscaling. Evidence
/tmp/gsp-directional-public-evidence-v1/summary.json/independentPHASEbeforecommit/
notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/scope/gates preserved.
PositiveautoChangingMagInteriorDAG/publicperiodic/tensornonlinear/nativequadglobal/
physics/wholeresources/crossover/Nscaling/populations/currentfourrepo/atleast2FINAL
independentaudits remainrequired. No production qualification.

### Guarded automatic DAG integral statistics reuse completed

Two primary changes bind planner statistics identity/version/mode and borrow its
existing CUDA physical-tree integral statistics after owner validation. No second
owner statistics cache. Unguarded traversal and integral/map/transfer/M2L/FFT/near/
complete action/public solve kernels unchanged. Inverse entire two primary modules
byteexact23457af1; other394 production modules byteexact. No covered PASS replay.
Collector /tmp/gsp-directional-integral-statistics-reuse-v1 CLOSED3NEW PASS;
XML30.128s/runner31.344613726250827s/rc0/402stable fullhelperclosure;
original87673drained/observed2400148+2400310absent/installedCore/GPU1.
Actual3sphere90edge exterior all65physicalnodes vsuncached rtol0/atol0 numerical
exact; MP100 allnodes+all10actualrootmaterialbounds maxrelative3.799433498426853e-15
<1e-11/abs0. 64queries identicalplannerarray/aggregationforbidden/map3918960B
unchanged/no2ndcache; externalplanner1560B countsaggregate/notwholepeakclaim.
Statistics identity/mode savedsource/receipts/query guards and version savedreceiver
backward guard PASS. Actual30edge magneticallnear15nodes/emptyDAGcontexts/nonempty
CUDAstats/uncachedexact/0ownedmapbytes/external360B/modeguard PASS; no positive
magneticinteriorFAR qualification. Exactcomparison notgeneraldeterminism orbyteproof.
Evidence /tmp/gsp-directional-integral-statistics-reuse-evidence-v1/summary.json;
independent PHASE audit beforecommit/notFINAL. No measured speed/wholeresource/
crossover/Nscaling qualification. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/scope/gates
unchanged. PositiveautochangingmagneticDAG/globaloriginalresidual/periodic/tensor/
nonlinear/nativequadphysics/whole resources/crossover/Nscaling/populations/current
fourrepo/atleast2 independentFINALaudits remainrequired. No production qualification.

### Complete directional retained dependency inventory completed

Primaryfmm_capacity appends two nonnumerical inventory functions; originalmodule
prefixbyteexact42896b3/395otherprodexact/componentreceipts/numericalkernels unchanged.
Completeactualphysicalinputs/preparedgeometry/alltrees/rootplans/boundDAGcarriers/
paths/stats/allmapowners inclfar.targets+origins/caller carriers/sharedlayouts/M2L/
originalnear/embeddings included. Actualdevice+backingpointerdedup/fullviewstorage;
disjointattributedcategories/overlappingstandalone. No cachedcopy orphysicsrecompute.
Workspaces/callercurrents/Rieszsolver/allocatorlibrary/host excluded/notpeakclaim.
v1CLOSED2NEWPASS1FAIL57.303s/401stable/96777drained/2416566+2416773absent;
leaf-onlyNestedlayoutNone fixedafterterminal; onlyfailedv2CLOSED1PASS7.278s/401stable/
7681drained/runner2449741absent/pytestPIDhonestlynullquickcompletion/childwaited.
IndependentpreliminaryP2 DAGcaller_carriers_missing preservedandfixedafterterminal;
only2NEW guarded+unguarded independentoversizedcarrierstorage controls v3CLOSED2PASS/
108.685s/401stable/66590drained/2763199+2764071absent/installedCore/GPU1.
Independentactualreachableobjectwalk equality automatic3sphere90edge4regions
4416024B/2893storageidentities; Nested2body12edgesalternatecouplingowner+sharedlayout
1039904B/186; unguardedcaller4406648B/2742/guarded4416360B/2897 withfull192+48+48+48B
backingstorage inclnonempty+emptyviews. Dense518400B/9216B respectively: no crossover.
Exactv1+v2primary andoriginaltest snapshots retained; inverse2fixes wholeprevious
modules exact/original3testprefixexact/no oldcollectorhash relabelling/no PASS replay.
Evidence /tmp/gsp-directional-retained-inventory-evidence-v1/summary.json;
finalindependentPHASE audit beforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN;
oldfailures/finalscope/gatesunchanged. Fullnativequadphysics/globaloriginalresidual/
positiveautochangingmagneticDAG/periodictensornonlinear/whole resources/crossover/
Nscaling/populations/currentfourrepo/atleast2FINALindependentaudits required.
No production qualification.

### Shared directional action validation traversal completed

PrimaryFFT extracts unchangednumericalbody behindoriginalstandaloneentryguard;
primaryDirectionalRWG uniquecompleteprototype/targets/origins ownerboundary checks
plusalltranslationtable/material/context/modeguards, then internaldirectnumeric
translationdispatch. FAR/systementry+exit/native savedcurrent guards retained.
No validationcache/newtensor/formula/workspace change. Inverseentire2primarymodules
byteexactd95e793/394otherprodexact. Actual10FAR/1owner boundary24→1 visits only;
no timing/scalingqualification. PublicstandaloneM2L fullcontextguards unchanged.
v1CLOSED5PASS1FAIL/XML471.841s/runner473.39361087139696s/404stable/54134drained/
2862959+2865011absent/installedCore/GPU1. Uncontrolled exact180cols FAIL106entries/
32400/maxabs6.308948006325748e-29/maxrel6.573500093894881e-05 preserved/notPASS.
Onlyfailednumerical+NEWportableoraclelifecycle v2CLOSED2PASS/XML705.607s/runner
707.3861014246941s/402stable/89329drained/3129032+3131424absent. Deterministiccolumns
exactunchangedrtol0/atol0 then previousenabled/warnflagsrestored; defaultsubsequent
15noncontigH/current/higher/Riesz/dot max9.485997359820047e-16<1e-11/saved0/
retainedinventorysame. Defaultelementwise determinism unqualified. No originalweak
matrixreintegrated/archive/no coveredPASSreplay. Fixture3bodies90edges/allmagintsnear.
Savedwhole+standaloneM2L statistics/path/table/context identity/mode guards and
statisticsversionafterexterior caughtbefore systemreturn PASS. Necessaryaffectedold
2CartesianFFTmaterial+1Nestedreverseowner failure/configcontrols PASS.
PreliminaryP2 tmporacledependency preserved; exactbaselineAST557fd963…3233c now
bundledrelative tests/sie/fixtures/directional_action_validation_legacy_far.py;
NEWavailability/SHA/names/compile/lifecycle PASS. Inversetestchanges restoresgenuine
v1snapshot/all396productionidenticalbothcollectors.4NEW+3necessaryold=7distinctPASS.
Evidence /tmp/gsp-directional-action-validation-evidence-v1/summary.json;
independentfinalPHASE auditbeforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/
oldfailures/scope/gatesunchanged. PositiveautochangingmagneticDAG/globaloriginal
residual/nativequadphysics/periodictensornonlinear/whole resources/repeatedtimings/
crossover/Nscaling/populations/currentfourrepo/atleast2FINALaudits required.
No production qualification.

### Configured native directional carrier catalog completed

Primarypublicconfig/region remove hard32 bound while retaining typed configured capacity/default32;
DAG reachable catalog obeys root configured max_families, including deliberate lowerthan32 closure constraint.
Shared Sparse/DAG projection nativeCUDA exactbit/order unique replaces pairwise duplicate loop;
signedzero/order distinct; successfulbits/degrees/keys released before mapcapacity;
4096+128C tensorcatalog allowance admittedbeforeunique, maxwithcontext allowance, library/allocator excluded/notmeasuredpeak.
Inverse3entireprimarymodules byteexactb6b949f/393otherprodexact; kernels/material/current/nearformulasunchanged.
Actual9closedtetra54edges72rootFAR/72root+DAGfamilies configured512;
all72rootMP100 max5.0230605398192025e-15<1e-11/abs0;
family71originalq16weak7.489663898904781e-13/H8.886222954509061e-13<unchanged1e-8;
15noncontig nativeforward/H/current/adjointcurrent/higher/dot max9.23843472041669e-15<1e-11/saved0.
Mapowned9570240B componentonly/notcompleteinventory/wholepeak/crossover.
v1CLOSED3PASS2FAIL/XML55.135s/runner56.680527531541884s/400stable/session53046drained/
3871275+3871452absent; originalmissingstopping+nonexistentmapinterception testfailurespreserved.
Onlytwofailedtestbodies fixed afterterminal; v2CLOSED2PASS/XML6.570s/runner8.169815125875175s/
399stable/session31640drained/runner3953504absent/quickpytestPIDnotobservedhonestNULL/childwaited.
396productionidenticalbothcollectors/passedtest+helper+fixtureASTunchanged;
failedcapacityfixtureonlyphysicalroot omitscoveredDAGmaps/no coveredphysicalPASS replay.
4NEW+1necessarychangedold=5distinctPASS/installedCore/GPU1both;
actualdefault32capacityreject/noFARrelabel/duplicatekeys/preuniqueworkspacerejectbeforemaps;
nativeunique signedzero/order+temporaryrelease lifecycle PASS.
Original3prelaunchP2 preservedinprelimv1/fixedbeforelaunch/clearprelimv2.
Evidence /tmp/gsp-directional-configured-catalog-evidence-v1/summary.json;
independentfinalPHASEauditbeforecommit/notFINALGoal. GoalACTIVE/GSP01/GSP09OPEN/
alloldfailures/finalscope/gatesunchanged. Completeglobaloriginalresidual/current/nativequadphysics,
positiveautochangingmagneticDAG/periodictensornonlinear/wholeresources/repeatedtimings/crossover/
allpointNscaling/populations/currentfourrepo/atleast2FINALaudits remainrequired. No production qualification.

### Original reciprocal scalar near packing completed

Primarypackednear new explicit reciprocal alignedsquare/free-space fixedscalar primitive;
allreversepair/JM/normalizedphysicalnode maps validated BEFOREdiscardinganyrecords;
canonicaloffdiagfirst/fullindependentdiagonals, reverse D/K byT notH;
sharedoriginalmaterial/adjoint/current kernel, actionbatchboundedreverseindexscratch,
no extra retainednumeric/cache/globalmatrix. Policy/count/id/version/mode/savedcurrent guards.
Inverseentireprimary byteexact56a937a/395otherproductionexact.
GPU1/installedCore originalsession66280 drainedrc0/235674+235878absent;
v1CLOSED5PASS/XML10.438s/runner11.845809794962406s/398stable/4NEW+1necessaryaffectedold.
Actual2closedtetra12edges externalinterleaved5/4/3rectanglemaps and6edge lossymagint2/2/2
samegroupdiagonal/reverse:9logical/6stored/6integrationcalls each;
all9ordered originalindependentD/K reference maxbalancedweak2.3571155598893304e-16/
originalH2.7138230498049454e-16<unchanged1e-8;15noncontignativeforward/H/current/
adjointcurrent/higher/dot max6.814520471709526e-16<1e-11/saved0.
Primitiveexterior6032→4072B/int2000→1352B exactcompactdisjointstorage/notwholeinventory/crossover.
Onebytebelowpreallocation/missingreverse/duplicate/JMmap/strictnodeoffsetrejectbeforefactory;
no-diagonal4reverse-consistentpairs inconsistentnormalizednode1mapsrejectprecanonical.
InitialstaticP2 namespace-mapbindingpreserved/fixedbeforelaunch/prelimv2clear.
No coveredPASSreplay/no oldfullmatrixreintegrated/archive.
Evidence /tmp/gsp-reciprocal-scalar-near-evidence-v1/summary.json;
independentfinalPHASEauditbeforecommit/notFINALGoal. Internalpackingonly; publicCartesian/
directionalconfiguration/capacity/factory/periodicexclusion/exactstate next. Original3840resources
failurespreserved; wholepeaks/repeatedtimings/crossover/allpointNscaling/globaloriginalcurrent/
residual/nativequadphysics/periodictensornonlinear/populations/currentfourrepo/atleast2FINALaudits
remainrequired. GoalACTIVE/GSP01/GSP09OPEN/alloldfailures/finalscope/gatesunchanged.
No production qualification.

### Public reciprocal scalar Cartesian FMM integration completed

`Scene -> Simulation.sie -> Result` accepts `FMMConfig(near_storage='scalar_reciprocal_dk_v1')`. Strict payload `cartesian_fmm_reciprocal_dk_config_v1` binds all explicit parameters. Existing default full-matrix and ordered scalar-D/K identities remain unchanged. The new policy is free-space scalar float64 geometry/complex128 currents only; tensor and periodic extensions remain required separately. Dense/compact/FFT tables and dense/separable shifts retain their existing semantics. Public Ewald/periodic and complex64 cases reject before setup.

The existing regional factory requires identical test/source physical basis and tree for reciprocal near storage, uses the original independent D/K integration callback and delegates to the primary PackedReciprocalScalarNearChannels. One off-diagonal direction is stored; reverse kernels use ordinary transpose, while diagonals stay full and Hermitian conjugation remains separate in the common adjoint. Whole action retained admission counts canonical near pairs, maps and CSR before allocating moment maps. Every non-near category is unchanged. The exact retained receipt excludes input geometry, Riesz/solver state and setup/action/backward workspace; it is not a whole peak certificate.

Inverse proofs recover all five primary modules byteexact to 6cf4638;391 other production modules, including the numerical packed primitive, remain byteequal. All396 production hashes are stable across both Maxwell collectors. Bench69 production modules are unchanged. New archive coverage uses existing native numerical/operator/RHS bindings, with no duplicate numerical implementation.

Original collector v1 CLOSED4PASS1FAIL (XML55.325s/runner56.895664491690695s). Its original 12-edge two-tetra integration fixture has only diagonal near pairs, so the compression assertion failed before numerical comparison. This coverage failure and the original tested test bytes are preserved. After the original handle drained and actual PIDs were absent, only the failed test changed and one new 60-edge two-sphere fixture was added; all passed test/fixture ASTs and production bytes are unchanged. Failed-only v2 CLOSED1PASS (XML181.116s/runner182.60391335748136s). No old PASS was replayed. Both399 frozen inputs remained stable. Their observed runner/pytest PIDs are absent and original handles drained.

The new full operator fixture uses two closed radius.01 spheres at x=+/- .5,300MHz, eps2/3, mu1.3/1.4, electric conductivity.01 and magnetic conductivity.02. Original near q3/singular3, order8/quadrature12, leaf2/separation.25, FFT/separable and existing budgets remain explicit. All120 columns of the original independently assembled full system pass balanced relative1.3300657565261127e-15 and original H1.4490868705064444e-15 against1e-8 gates. Native15-RHS forward/H/current/adjoint-current/higher/dot maximum6.08908883020928e-16 passes1e-11; saved action intermediates are zero. Exterior1458 logical near blocks become756 stored; interiors900 become465 each. Exact owned retained2,156,512B equals admission, versus ordered-D/K2,302,528B; only near304,680B becomes158,664B. The actual compressed case rejects capacity minus one before maps. Shared original near quadrature is not a convergence oracle; no physics or crossover qualification follows.

The original v1 public solve already passed and is reused once: two spheres60edges at +/- .2,90MHz, eps2/3 and mu1.2/1.3, q4/singular3, automatic order6..10/G1e-6/H1e-5/direct-first-wave, leaf8/separation.25/FFT/separable, original bounded16-edge primary factors. GMRES rtol1e-12/atol0/max240/restart120 converges in93 iterations. Compressed dual2.1070537409796903e-13 and original eight-edge sample3.020825082849042e-13 pass1e-8. The sample is not a global original residual certificate. Exterior242 becomes132 stored (46,920B), each interior64 becomes36 (21,640B). Exact currents/configuration restore with constructors and solve forbidden, and changed near/table/bound/shift policies reject.

Completed producer SHA256 0f7e0fa8e7a3f1079ca9eb7404dc6f266c3f4ce71d204f06221a6310179b3949 is reused by a new Bench control without another solve or producer save. Independent Scene/full numerical declarations precede loading. J/M archive bytes are exact; four changed policies reject in both native restore and Bench manifest validation. Native schema11 results remain unqualified candidates. Bench collector CLOSED1PASS/XML3.857s/runner5.340552922338247s with473 stable source/producer inputs. Its quick pytest PID was not observed live and is not invented; recorded runner absent, original handle drained, child waited rc0.

Six distinct PASS consist of four NEW Maxwell controls, one necessary affected old public scalar-scope control and one NEW Bench archive control. This is selected phase acceptance, not current full regression or FINAL Goal audit. Independent PHASE audit precedes commit. Shared GPU timings are not a performance benchmark. Original3840-edge crossover/peak failures remain preserved; whole setup/warmups/differentiable forward/backward/solver peaks, all-point scaling, directional reciprocal integration, periodic/tensor/nonlinear paths, populations, current four-repository final acceptance and at least two independent FINAL audits remain required. Goal ACTIVE; GSP01/GSP09 OPEN; original scope/failures/gates unchanged. No production qualification.

### Public reciprocal directional near storage integration completed

`DirectionalFMMConfig(near_storage='scalar_reciprocal_dk_v1')` selects the primary reciprocal original D/K near factory through `Scene -> Simulation.sie -> Result`. The strict full-field payload is `automatic_directional_reciprocal_dag_fmm_config_v1`. Default ordered-D/K payloads omit the new default field and retain `automatic_directional_dag_fmm_config_v1` byte identities. The public policy remains free-space scalar float64 geometry/complex128 currents with iterative no-fallback solves; tensor, periodic and nonlinear directional paths remain open requirements.

The primary structural helper validates native float64 physical terminal nodes, unique pairs and complete reverse closure, then counts canonical a<=b kernels, full diagonals, maps and CSR. Both automatic Sparse and DAG builders check reciprocal capacity before map or original integration factories. The complete regional factory separately checks exact full near/FAR coverage and identical physical basis/tree, then delegates to PackedReciprocalScalarNearChannels without changing FAR maps, tables, plans or global ideal bounds. Reverse D/K use ordinary transpose; Hermitian material/current composition remains independent. Added bounded reverse-index workspace is explicit and still excludes setup, allocator and backend library peaks. The near policy participates in saved-current regional guards and appears only for the new mode in receipts.

Five primary whole-module inverse proofs recover baseline8ecd696;391 other production modules, including all numerical FAR and packed D/K kernels, remain byteequal. Three Maxwell collectors retain the identical396 production hashes;69 Bench production modules are unchanged. Existing default configuration identity is the one necessary affected old control. No previously covered FAR/native-quadrature/full-original assembly or old solve is replayed.

The original collector CLOSED4PASS1FAIL/XML1817.850s/runner1819.5387367866933s/401 stable inputs. Its configuration test mistakenly used StoppingRule's row_norm default, rejected by the existing public FMM contract. The original tested source and failure remain preserved. Only the failed function changed after the original handle drained and actual708736/708896 were absent; all passed functions and the physical fixture retain exact ASTs and all production remains unchanged. Failed-only v2 CLOSED1PASS/XML3.268s/runner4.508753350004554s/406 stable inputs. The additional changed Sparse-builder negative control CLOSED1PASS/XML3.969s/runner5.425949591211975s/400 stable inputs, without constructing maps, original near or a positive Sparse FAR action. These two quick pytest PIDs were not observed live and are not invented; recorded runners absent, original handles drained and children waited rc0.

The actual three-sphere fixture has90 physical edges/180 currents, radius.02 at (i,1e-6*(i%2),0),300MHz, eps2/3/4, mu1.3/1.4/1.5, electric conductivity.01 and magnetic conductivity.02. Original q6/singular3/pair4/far16/nearmax32, leaf4/separation.15/cone2^20/quad12 and all existing budgets remain unchanged. Exterior has four positive changing-carrier links; all three magnetic interiors have zero FAR. Exterior507 logical near blocks become273 stored (74,864B); each interior169 becomes91 (24,960B). Exact retained complete native dependency inventory4,416,024B becomes4,292,472B, saving123,552B. This inventory includes shared inputs/plans and excludes workspace, solver/Riesz and allocator; it is not a whole peak or crossover certificate.

All180 complete columns compare the new reciprocal action with independently integrated original ordered D/K near blocks and the identical shared FAR owners. Balanced relative1.5446254372149055e-16 and independently integrated ordered-near H5.258060311068895e-16 pass unchanged1e-11 comparison gates. Native15 noncontiguous RHS forward/H/current/adjoint-current/higher/dot maximum7.656287059748157e-16 passes1e-11; no internal tensors are saved by current backward. Changed near policy rejects receipt and saved backward. This controlled composition comparison does not repeat or certify original full-matrix FAR quadrature or physical convergence. Its explicit shared-FAR limit is retained in test and evidence metadata.

The public GMRES solve uses unchanged original bounded16-edge factors, rtol1e-12/atol0/max240/restart120 and converges in115 iterations without fallback. Compressed dual1.115405123955778e-12 and original eight-edge sample1.5741683304210806e-12 pass1e-8. The sample is not global original residual/current error/physics certification. Exact currents and configuration restore with constructors and solve forbidden; near/candidate/phase/cone policy changes reject. Native physical schema11 stores the physical result, not an executable compressed operator.

Producer SHA256 aa0937fa56d6dd7291ce872dfefe65f87d65e15f256297e7a4c9abb45301ff24 is reused once by the independently declared Bench Scene/full configuration. J/M bytes are exact; the four policies reject both native restore and the independently bound Bench manifest. No producer save, assembly or solve is repeated. Bench CLOSED1PASS/XML4.078s/runner5.825879503972828s/473 stable inputs. Its quick pytest PID is honestly unobserved; runner770814 absent, original handle51596 drained, child waited rc0. The archive remains an unqualified candidate.

Seven distinct PASS are five NEW Maxwell controls, one necessary affected old configuration control and one NEW Bench archive control. Independent PHASE audit precedes commits, without numerical reruns. This is not current full regression or FINAL Goal audit. Original3840-edge reciprocal resource collector is separately LIVE on its frozen8ecd696 provider; no source, order, budget or failure is changed here. Its resources and all previous failures remain separate. Positive automatic changing-carrier magnetic FAR, general tensor/periodic/nonlinear paths, converged quadrature/native/global physics, whole setup/actions/backward/solver resources, all-point scaling/crossover, populations, current four-repository final acceptance and at least two independent FINAL audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN; original scope/failures/gates unchanged. No production qualification.

### Native reciprocal compact translation primitive completed

The primary `CompactElectromagneticM2L.reversed_translation()` returns the opposite free-space displacement action while borrowing the original coefficients, indices and binomial weights. Radial Taylor coefficients change by degree parity under r -> -r; generated Green tables apply that parity and gradient tables apply one further minus. Complex wave number and electromagnetic material factors are never conjugated by displacement reversal. The existing Hermitian action independently conjugates the full material/kernel algebra. No reverse coefficient vector, dense retained table or numerical layout is allocated by construction. The reverse receipt distinguishes shared dependency bytes from zero additional owned tensor bytes.

Owner identity, fixed material/configuration/tensor identity and versions, plus borrowed requires_grad modes participate in reverse guards. Saved-current backward re-enters the guarded original and reverse actions. Reverse-of-reverse construction and differentiable coefficient borrowing fail closed. Existing default state, receipt and numerical branches remain unchanged; one complete primary-module inverse proof recovers870d0eed, and395 other production modules remain byteequal. Public FMM configuration, topology, regional assembly and capacity planners have not been changed by this internal primitive stage.

The original GPU1/installed-Core collector CLOSED6PASS/no skips/XML5.003s/runner6.511025075800717s with399 stable sources. Five NEW controls comprise orders0/3/6 against the independent original dense factory at the opposite displacement, MP100 analytic radial derivative samples, and saved owner/mode/version guards. One necessary affected original compact order0 control passes. The short child completed before live PID observation; its pytest PID is honestly NULL. Execution runner1048008 is captured, original handle3042 drained rc0, outside ps confirms runner absent and the runner waited its child rc0. No previous complete FMM, solve or resource control is rerun.

Lossy magnetic scalar medium uses eps1.5/mu1.3, electric conductivity.02/magnetic.03 at300MHz and displacement(1,.3,-.2). All generated table entries agree with opposite-center original dense tables: max Green relative0, gradient7.128781524525757e-17. Fifteen noncontiguous native RHS per order pass forward/H/current/adjoint-current/higher/dot at1e-11, maximum2.5930417383723824e-15, with no saved internal tensors. MP100 independently differentiates exp(ikr)/(4*pi*r) at both centers for nine degree0-3 samples, max relative1.0179553828707576e-15<1e-11. Unique two-direction retained dependencies are104/2656/11760 bytes, exactly one original coefficient/layout inventory rather than208/5312/23520 independently owned bytes. Constructor forbids Taylor/layout generation and changes no live CUDA allocation; this is not a full setup/action peak bound.

Independent read-only PHASE audit precedes commit. This primitive does not yet enter public FMM construction/capacity, and it does not certify full assembled RWG errors, native quadrature, physical convergence, high-order/full-system resource peaks, solver resources, scaling or crossover. The separately completed original3840-edge reciprocal-near resource result remains bound to its frozen8ecd696 provider; its full allocator peak failure is not closed by this primitive. All original failures, scope and gates remain unchanged. General tensor/periodic/nonlinear paths, positive automatic magnetic changing-carrier FAR, all-point resource/complexity acceptance, Bench populations, current four-repository acceptance and at least two independent FINAL Goal audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN; no production qualification.

### Original reciprocal-near full GPU resource measurement completed

One NEW original fixed-ka resource control completed on the frozen8ecd696 provider. The two spheres retain radius.01 at x=-.5/+.5, subdivisions3/3840 physical edges, vacuum exterior/eps2 mu1 interior,300MHz, admitted p11, positive_tail_min_v1 G1e-6/H1e-5, leaf30/separation.25, original q4/singular3, compact tables/separable shifts and all original1GiB/2GiB budgets. The completed original order plan and ordered-DK resource are hash-pinned; automatic replanning is forbidden. Only near storage selects scalar_reciprocal_dk_v1. All396 production modules equal8ecd696; the new test borrows only the existing original whole-interval measurement helpers, whose complete source is frozen. No old test, original all-column reference, solve or producer save is replayed.

Original handle77232 drained rc0; actual observed runner665864/pytest666067 are absent outside the sandbox. Collector CLOSED1PASS/no skips/XML4137.406s/runner4138.9490486979485s/412 stable frozen sources. Installed Core, Torch2.13.0+cu130, mpmath1.3.0, RTX A6000/GPU1 execution identities are captured. The test PASS covers faithful measurement, structural equality, native finite actions and fixed-current derivative gates; it does not mean resource crossover PASS.

Owned retained tensors equal the complete structural plan795,719,008B versus original ordered-DK1,051,291,168B and same-N full complex128 dense943,718,400B. All non-near categories remain exact: table354,688,320B, leaf moments178,913,280B, shifts1,908,384B, with the same tree/center/map/body categories. Near259,874,344B replaces515,446,504B, saving255,572,160B exactly. Exterior22,050 logical blocks become11,130 stored/126,173,720B; each interior58,136 becomes29,192/66,850,312B. The retained-only comparison falls below the dense storage number at this one point, not an all-point production crossover.

Two native RHS pass full forward/H dot9.583499407993958e-16 and actual current-gradient6.380181322921783e-17 at unchanged1e-11. The allocator interval begins before construction and includes setup, all warmups, actual forward/H, differentiable forward, saved-current backward, peak-reset gaps and final finite checks. Whole peak allocated965,273,088B and reserved1,048,576,000B both exceed the unchanged same-N dense storage943,718,400B: the resource peak gate remains FAIL. Prior ordered-DK peaks1,224,886,272B/1,317,011,456B are preserved. The setup allocated934,558,208B alone would miss the later actual backward peak and cannot be used as a full certificate. No allocator reset, omission of warmup/differentiable forward, loosened gate or retained-only substitution is made.

Setup1695.65195s; warm forward406.51509s/H411.42072s; measured forward400.54230s/H402.51946s; differentiable forward387.51176s/backward396.79543s. These shared-GPU timings are observations, not isolated performance or N-scaling certification. Absolute current-process allocator peaks include provider/caller allocations live in the fixture; retained action capacity excludes provider geometry, solver/Riesz and allocator workspace. The interval has no solver or teardown and is not an all-device/external-library resource certificate.

Later incoming main8acdafe source applicability is recorded separately, never relabelled as the tested source:390 modules are byteequal; five directional-only changes and one compact opposite-displacement extension have complete whole-module inverse proofs. This original Cartesian resource path never calls reversed_translation; its original compact coefficient/layout/material/action and default state/receipt branches remain unchanged. Independent PHASE audit checks that chain and the planned append-only merge. No production code changes in this resource evidence stage. The isolated8ecd696 test/docs commit is integrated onto8acdafe by preserving both earlier phase entries and appending this resource entry once; all396 incoming main production hashes remain unchanged.

The new measurement reduces but does not close resource peak/crossover failure. Original GSP01/other physical failures remain unchanged. Whole solver resources, all-point fixed/growing/high-frequency scaling and crossover, tensor/periodic/nonlinear paths, positive automatic magnetic changing-carrier FAR, population acceptance, current four-repository full acceptance and at least two independent FINAL Goal audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN; no physical, complexity, resource or production qualification.

### Public exact-displacement shared compact FMM integration completed

`FMMConfig(table_storage="compact_shared_displacement_v1")` selects exact same/opposite native displacement sharing within each physical region, medium and uniform expansion order through Scene/Simulation/Result. Strict full-field payload `cartesian_fmm_shared_compact_config_v1` binds the new table representation and all near policies. Existing default/compact/FFT/near-reciprocal payload identities remain unchanged. Public configuration and internal periodic system reject this new free-space mode for periodic kernels; native float64 geometry/complex128 currents are required. No tensor or nonlinear FMM qualification is added.

The primary catalog processes at most32pairs per geometry/key tensor. CUDA float64 center subtraction and exact IEEE integer keys choose one representative for equal or opposite displacement; signed-zero coordinates normalize for key identity only, while a1ULP displacement difference never merges. Only integer bits, orientation and topology transfer to host control. No approximate bins or additional numerical retained cache exist. Logical FAR pairs, order, physical projection/context and traversal order remain intact, including partitions without physical reverse closure. All original ordered separation/kernel-error/RWG bounds remain independently checked. Representatives generate original compact coefficients in batches at most32; reverse uses the already verified guarded degree-parity primitive. Each original/borrowed owner preserves material, identity, version, mode and saved-current guards.

Capacity and both plain/RWG order planners count actual distinct coefficient/layout storage before numerical setup, rather than logical-pair bytes. All ordered kernel bounds and RWG accumulation remain unchanged. Region checks cumulative table capacity before coefficients; system checks whole owned capacity and new table budget before moments. Native retained storage equals exact structural planning, with shared tensor pointers counted once. Near original integrals, maps, diagonal values and all existing electromagnetic/current kernels are unchanged. Nine complete primary-module inverse proofs recover638a2a89;387 other production modules, including compact parity/math kernels, remain byteequal.69 Bench production modules are unchanged.

Independent prelaunch review preserved and fixed two P2 findings before any numerical launch: shared admission initially kept all delta views and stacked all representatives; it now retains no such views and rebuilds<=32representative displacements per coefficient batch. A batching test imported a module fixture from a different test module; it was moved into the original module with all prior function/fixture ASTs unchanged, ensuring one complete near/reference construction. Prepared v1/v2 collectors were never executed. Original review snapshots and SHA proofs remain available; v3 prelaunch audit has no P1/P2.

Original v3 collector CLOSED6PASS1FAIL/XML335.830s/runner337.7495450126007s/411 stable frozen sources. The configuration negative omitted the established local_rwg_v1 Riesz declaration and failed before the intended periodic rejection. Original tested source, traceback and six successful controls remain unchanged. Only the failed function added that required setting after original handle8009 drained rc1 and observed1563920/1565056 exited. Failed-only v4 CLOSED1PASS/XML3.269s/runner4.513600739650428s/404 stable inputs. Its quick pytest PID was unobserved and remains NULL; execution runner1872351 absent, original handle69794 drained rc0 and child waited rc0. No successful physical/native/planner/public control was rerun.

The actual controlled system has two radius.01 spheres at x=-.5/+.5,300MHz,60 physical edges/120 currents, eps2/3, mu1.3/1.4, electric conductivity.01 and magnetic.02, original q3/singular3, p8/q12, leaf2/separation.25 and unchanged kernel budgets1e-6/1e-5. Exterior has two logical FAR pairs and one shared owner; both magnetic interiors have zero FAR. All120 complete columns compare independently generated original compact FAR tables plus identical near/projection objects: balanced relative0. Native15 noncontiguous forward/H/current/adjoint-current/higher/dot maximum1.5498535897101087e-16<1e-11, with zero saved internal tensors. Owned2,124,984B replaces2,148,480B; only M2L changes46,992B->23,496B. Capacity-minus-one/action and table budgets reject before moments; saved original-owner mutation rejects receipt/H/backward. This controlled sharing comparison reuses near integration and is not a new full original RWG or physics certificate.

The same physical leaf nodes supply729 actual cross-body logical couplings and462 distinct displacements. Every representative generation call is<=32 (fourteen32-pair batches and one14-pair batch); actual unique retained10,855,152B equals exact planned storage and budget-minus-one rejects before coefficients. No second near integration or complete solve is needed for this control. Both order planning branches use a single fixed p8 diagnostic, preserving every ordered contour/RWG result while changing only table bytes; this does not repeat the original global order search or certify implemented floating-point truncation.

The public two-body radius.01 x=-.2/+.2 eps2/3 mu1.2/1.3 scene at90MHz uses unchanged p6-10 admission, original q4/singular3, leaf8/separation.25, primary bounded16-edge factors, GMRES rtol1e-12/atol0/max240/restart120/no fallback. Selectedp6 converges in93 iterations. Compressed dual2.1372545739545198e-13 and original eight-edge sample2.88799511817202e-13 pass unchanged1e-8. The sample is not a global original residual/current-error or physical certificate. Public retained1,013,696B includes11,760B M2L. Physical Result schema11 saves once and restores exact currents/configuration with constructor/solve tripwires and near/table/kernel/shift tamper rejection. Its allocator stats were not reset over the whole test interval and are not resource evidence.

The completed producer SHA7231028437f24cb7a3e2b7e9cd41c8467f5be472eeba6cf1734e5f168250bfaa is reused once by Bench under independently declared physical Scene/full numerical identity. Exact J/M bytes, native schema11, unqualified-candidate role, constructor/solve refusal and the four native/manifest tamper rejections pass. Bench CLOSED1PASS/XML4.574s/runner6.0688778180629015s/473 stable inputs; actual observed1872942/1873810 absent and original handle95412 drained rc0. No producer solve/save is repeated. The original producer aggregate FAIL is not relabelled PASS; its successful public subtest/artifact is explicitly selected.

Eight distinct PASS comprise six NEW Maxwell controls, one necessary affected old configuration and one NEW Bench archive. Independent final PHASE audit precedes separate commits. Original3840 reciprocal-near allocator peak965,273,088B/reserved1,048,576,000B still exceeds same-N dense943,718,400B; this small shared-table stage does not close that failure. A changed original-resource measurement is separately prepared but has not launched or passed here. Native quadrature/global physical convergence, positive automatic magnetic changing-carrier FAR, tensor/periodic/nonlinear capabilities, full solver/setup/action/backward peaks, all-point scaling/crossover, populations, current four-repository full acceptance and at least two independent FINAL Goal audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN; original scope/failures/gates unchanged; no production qualification.

### Complete scaling resource interval workflow completed

The primary measurement workflow now records a continuous current-process CUDA allocator interval from before construction through all warmups, measured forward/H, differentiable forward, first current backward, reset gaps and final finite checks. Each reset first captures any preceding transient peak. Setup includes original geometry/provider construction; current differentiation reuses the measured H instead of computing a second H action. Every native action must be finite and meet the original1e-11 adjoint/current-gradient gates. An explicit complete-interval v2 plan chooses shared compact displacement tables, separable shifts, reciprocal scalar near and positive-tail admission. All27 pre-existing plan values except version remain AST-identical: two size families, four subdivisions, three repetitions, physical geometry/materials/frequencies, kernel/reference thresholds and original budgets are unchanged. The subsequent small original-reference control remains separately scoped outside this interval; no whole solver or teardown resources are claimed.

The primary summary includes differentiable-forward and continuous allocated/reserved peaks in every complete-family fit. It rejects missing warmups, raw native-action errors at or above1e-11, understated interval peaks, mismatched closed failure/source identities and incompatible resumed worker records. It continues to refuse fits for any incomplete or failed family; no tail selection or asymptotic/crossover qualification is introduced. Process-failed records now preserve their source hash.

Independent v1 static review found one P2: an affected existing negative still expected the previous source-hash error text although stricter early identity validation rejects first. The entire original test snapshot and v1 report/collector inputs are preserved. Only that negative regex changes to identity, with an exact whole-file inverse proof. Prepared v1 was never launched; v2 static review closes the finding before numerical execution. All396 production modules remain byteequal to mergedf5a80a7.

The one-shot v2 collector CLOSED23PASS/XML5.980s/runner7.558053915388882s/412 stable inputs:14NEW controls and9 necessary affected old workflow checks. Only one new actual GPU smoke runs fixed_ka/subdivision0/repetition1,60edges/120currents/300MHz/selectedp6 with original2RHS seed1729. It performs no original matrix assembly, solver or archive, and does not launch the24-case series. Actual owned864,544B equals planning, M2L11,760B; adjoint2.267006868810647e-16/current-gradient0 pass1e-11. Whole allocated92,140,544B and reserved138,412,032B exceed same-N dense230,400B: those point resource comparisons are FAIL, faithfully retained rather than converted from the23 test PASS. Backward increases reserved peak above setup104,857,600B, directly demonstrating why setup alone is insufficient.

Original session77366 drainedrc0; actual observed runner2560660 and pytest2562071 exited outside the sandbox. Installed Core, nativeCUDA float64/complex128 and GPU1 are recorded. No full regression, successful producer or covered physics control was replayed. The independent original3840 shared-table resource collector remains LIVE on its frozenf5a80a7 provider and is neither changed nor relabelled by this development-script stage. Its result is pending; previous3840 reciprocal-near peak FAIL and all original physical failures remain preserved.

This is a workflow/PHASE checkpoint, not an all-point resource series or FINAL Goal audit. Full solver resources, native quadrature/global physical acceptance, positive automatic magnetic changing-carrier FAR, tensor/periodic/nonlinear combinations, preregistered all-point scaling/crossover, populations, current four-repository acceptance and at least two independent FINAL audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN; original scope/gates unchanged; no production qualification.
### Bounded complete original weak residual diagnostic completed

The primary `operator_error.py` adds an opt-in scalar free-space diagnostic which streams every configured original J/M weak row with independent positive target/source batch limits. It calls the existing primary `ScalarSystemBlocks.sampled_action` and scaled norm primitives, constructs no global dense matrix and uses no compressed action. Every supplied RHS has its own native absolute/relative gate; noncontiguous multidimensional batches preserve their output shape. Zero RHS with zero error passes, while nonzero error at zero RHS has a JSON-null relative error and fails the unchanged zero absolute tolerance. The default relative gate is1e-10; the caller's finite nonnegative absolute tolerance is explicit. Invalid/empty/nonfinite/device/dtype inputs fail before original action.

The diagnostic binds input versions/modes, the original provider/configuration, all numerical scalar medium fields and body routing/identities, and rechecks them at target-batch boundaries and completion. Independent prelaunch v1 review found a P2 in identity-only configuration/material guards; the prepared v1 collector was never launched. The fix and new same-object configuration/material/body mutation controls are frozen separately before any GPU execution. The entire prior production module is unchanged when the appended function is removed;395 other production modules remain byteexact89621e7. No existing solver residual gate or default solve behavior changes.

This measures the complete supplied vector residual for the configured original quadrature. It does not measure error over all operator coefficients, certify quadrature convergence/continuum accuracy/rounding, qualify peak resources or grant production status. Selected scalar controls do not admit tensor or periodic providers. GSP-01 physical failures remain OPEN; the finer resonant geometry still requires compressed capacity admission and new source-bound physical acceptance rather than a dense allocation larger than the GPU memory. Full resource/scaling, tensor/nonlinear/periodic combinations, populations, current four-repository acceptance and at least two independent FINAL audits remain required. Goal ACTIVE; original scope and thresholds unchanged.

The one-shot new collector CLOSED13PASS/XML8.889s/runner10.790005689486861s/406 stable inputs. One module fixture constructs two actual disjoint lossy magnetic tetrahedra,12 edges/24 currents and one independent dense original reference. Native original controls cover15 noncontiguous RHS, exact target2/source3 physical-edge block enumeration,1e-12 native independent RHS norms, separated1e±120 currents and an error at original edge10 outside the old default8-edge sample, zero/nonzero action at zero RHS and no saved autograd tensors. Cached original controls test the scalar/explicit absolute gate, invalid arguments, input version/mode, provider/configuration/material/body mutations and nonfinite action without replaying original quadrature. Only new tests ran; no old PASS regression, physical producer solve or archive replay. Original session52414 drainedrc0 and observed runner3197581/pytest3197938 are absent outside the sandbox. Installed Core/nativeCUDA/GPU1 are recorded.

### Original shared compact fixed-ka full GPU resource measurement completed

The original two-sphere3840-edge fixed-ka experiment now measures exact native displacement sharing together with the already integrated reciprocal original scalar D/K near storage. Radius0.01m/centers±0.5m/subdivision3/eps2-vacuum/300MHz/p11/leaf30/separation0.25/q4-singular3/positive_tail_min_v1, G1e-6/H1e-5, all original retained/table/near/projection budgets and native two-RHS seed1729/1e-11 gates remain unchanged. Completed original p11 order admission and prior reciprocal-near resource receipts are SHA-bound and reused, never re-searched or replayed. Only FAR table ownership changes; logical physical interactions and original near receipts remain exact.

Actual owned retained storage585,948,608B equals the primary structural plan, down from795,719,008B by exactly209,770,400B of table savings. M2L144,917,920B, near259,874,344B and projection180,911,744B are separately recorded. All6738 logical FAR couplings remain;2753 unique coefficient owners share their exact same/opposite displacement storage. All other capacity categories and reciprocal-near receipts are unchanged. No geometry/coefficient approximation, threshold relaxation or interaction removal is used.

The complete current-process allocator interval includes original construction, every forward/H warmup, measured forward/H, differentiable forward, first current backward, peak-reset gaps and final current/finite checks. Peak allocated751,039,488B and reserved803,209,216B are BOTH below unchanged same-N dense943,718,400B. This selected point comparison is PASS. Previous reciprocal-only peaks965,273,088B/1,048,576,000B and ordered-D/K peaks1,224,886,272B/1,317,011,456B remain preserved FAIL evidence, not overwritten. Native adjoint3.8408795747260305e-15 and current-gradient7.12969985810129e-17 pass1e-11.

The original collector CLOSED1NEWPASS/XML4297.158s/runner4299.230414410122s/417 stable inputs. Original session66501 drainedrc0; observed runner2055404/pytest2055588 are absent outside the sandbox. Installed Core/nativeCUDA/GPU1 are recorded. Timing is from a shared busy GPU and is not a performance claim. No solver, original all-column reference, repeated producer archive, full regression or all-point scaling series runs in this measurement.

Frozen tested baselinef5a80a7 remains explicit. Current92806ac source applicability is separate:395 primary modules byteexact, and removing only the appended unused complete-original-residual diagnostic restores the entire old operator_error module. The intervening89621e7 development-workflow phase changes no production module. Never relabel this collector as numerically tested at the newer revision.

This is one original fixed-ka setup/actions/current-backward resource point, not a whole solver resource certificate, scaling/crossover series or FINAL Goal audit. Growing-ka3840 and other original collectors remain LIVE on frozen providers. Full solver/workspace/host resources, every preregistered size/repetition, general/tensor/periodic/nonlinear native quadrature/physics, production populations, current four-repository acceptance and at least two independent FINAL audits remain required. Goal ACTIVE;GSP01/GSP09 OPEN;original scope/gates unchanged;production qualification false.

### Bounded original scalar RHS integration for compressed runtimes completed

The primary SystemOperator RHS dispatch now uses the existing selected original analytic lifting integrator for every local edge in free-space scalar hmatrix/H2/direct-H2/Cartesian-or-directional-FMM preparation. The integration primitive bounds classification by the declared pair/far batches and retains no global face-pair classification array. All local J/M rows, body order, analytic plane sources, integration/near-refinement policies and material differences remain. Dense, periodic and dyadic configurations keep their prior dispatch. The numerical lifting/kernel/quadrature primitives are reused, not copied or replaced.

The primary selected integrator now retains its existing near-refinement observations instead of discarding them: maximum order, pair evaluations and maximum accepted relative difference, together with the same near norm/estimator and independent touching/far convergence requirements. This is additive receipt metadata; no RHS global quadrature, continuum or production certificate is claimed. No solver residual threshold, public configuration default or constructor/solve-free archive contract changes.

Original v1 collector CLOSED3FAIL4PASS/XML129.946s/runner131.47759534232318s/406 stable inputs. All three failures were invalid test-helper stopping configurations before the new native RHS comparison/dispatch: direct-H2/FMM rejected the existing row-column-norm default. The whole original test and failure receipts remain preserved. Only the helper selects legal none preconditioning for those two backends; H/H2 helper defaults, all other test/fixture AST and all396 production bytes remain identical to the original numeric run. Independently reviewed failed-only v2 CLOSED3PASS/XML4.458s/runner6.0029570842161775s/420 stable inputs. The four previous successes are not replayed.

Seven DISTINCT targeted controls pass: sixNEW and one necessary affected selected-row ordering/validation control. Actual two disjoint lossy magnetic tetrahedra with two nonparallel analytic plane sources give complete J/M RHS relative difference3.66681773557313e-16 against the original integration, below unchanged2e-12. Per-body classification maximum11 and exact pair coverage are checked. Four cached-original route controls cover H/H2/direct-H2/FMM without replaying quadrature. A necessary actual public two-body shared-compact FMM prepare/run passes solve gates, bounded RHS and retained near-estimator checks; the new complete original weak residual diagnostic independently measures all120 rows with target19/source17 and gives1.3466771206667718e-13 against unchanged1e-10/atol0. No global original matrix or producer archive is constructed by this control. The public successful solve is not rerun in v2.

Original v1 session11831 drainedrc1 and observed runner3505483/pytest3505735 exited; v2 session26428 drainedrc0 and observed runner3602092 exited. Its short-lived pytest PID was not observed and remains explicit null; runner child-wait receipt/XML supply terminal evidence. Installed Core/nativeCUDA/GPU1 are recorded. No covered configuration suite, full regression, physical predecessor or archive is replayed.

Baseline92806ac and incoming54ecd1e remain explicit.394 other production modules are byteexact; runtime complete old-module inverse removes only the new scalar compressed dispatch; lifting entire prior prefix and every other numerical function remain unchanged. Primary reverse-patch validation binds the selected metadata-only near observation change. The stage does not qualify general tensor/periodic/nonlinear material/source gradients, full quadrature/physical convergence, RHS asymptotics or complete setup/solver GPU resources. GSP01/GSP09 remain OPEN. Original f30 capacity/growing3840/other collectors retain their frozen providers and original handles; no LIVE source change. Goal ACTIVE;final scope/gates unchanged;four-repository final acceptance and at least two independent FINAL audits remain required;production qualification false.

### Original resonant f30 compressed capacity preflight completed

The failed-physics successor preserves the original resonant sphere radius0.1m/frequency414635862.3319615Hz/epsilon12+0.01j, frequency-mesh30 with27000edges/54000currents, q4/singular6/pair64/far8192/near1e-5/max16, physical0.005 and complete original-vector residual1e-10/atol0. Historical f24 physical failures remain SHA-bound and are not replayed. No predicted finer-mesh accuracy is reported as measured.

Native GPU1 geometry/provider/two independent physical support trees feed the existing primary capacity planner. Same trees/partition, leaf30/separation0.25, reciprocal scalar D/K near, exact displacement-shared compact tables and separable shifts are retained. Minimum p6 requires14,829,648,416B against unchanged2GiB action budget: M2L11,473,832,160B and near3,062,449,136B each exceed unchanged1GiB component budgets; projection291,371,328B passes. For this exact partition/storage, increasing p6..12 cannot repair the minimum capacity rejection. No kernel order search, expansion, original quadrature, compressed action, solve, backward, dense global matrix or field acceptance is launched. This is a configured candidate capacity rejection, not a claim that every compressed backend is impossible.

The preflight itself records peak allocated70,482,944B/reserved92,274,688B, limited to geometry/provider/tree/capacity planning. These peaks are not setup/actions/solver or whole-resource qualification. Same-N dense matrix alone would require46,656,000,000B; it is not constructed. One NEW preflight instrumentation control PASS/XML2689.929s/runner2691.5254482720047s with411 unchanged frozen inputs records the rejection; its pytest PASS grants no physical, resource or production qualification. Original session37714 drainedrc0; observed runner3362671/pytest3363646 are absent outside the sandbox. Installed Core/native CUDA/GPU1 remain explicit.

Frozen tested baseline92806ac remains explicit. Current d06eed1 applicability is separate:394 other primary modules exact; two subsequent runtime/lifting RHS changes are unused by this geometry/capacity-only test and documented by the preserved continuation proof. No old PASS/full regression/archive or numerical rerun. GSP01/GSP09 OPEN;physical successor solve awaits a newly preregistered admissible compression strategy under the same original gates. Goal ACTIVE;full scope, current four-repository final acceptance and at least two independent FINAL audits remain required;production qualification false.

### Original shared compact growing-ka full GPU resource measurement completed

The original two-sphere3840-edge growing-ka resource experiment preserves subdivision3/radius0.01m/centers±0.5m/eps2-vacuum/2.4GHz/p12/leaf30/separation0.25/q4-singular3/direct_first_wave_min_v1, G1e-6/H1e-5, all original2GiB action/1GiB table-near-projection budgets and native two-RHS seed1729/1e-11 gates. Closed original p12 admission and original full-matrix compact resource receipts are SHA-bound and reused, with no repeated order search or historical producer archive. Reciprocal original scalar D/K near and exact native displacement-shared compact tables preserve the same physical logical interactions and all other retained categories.

Actual owned668,235,432B equals the primary structural plan, compared with original full-matrix compact1,656,499,936B. Near259,874,344B saves725,095,104B; table181,808,120B saves263,169,400B. Unshared reciprocal compact intermediate931,404,832B isolates the two storage savings. Projection226,308,368B and every other category remain exact. All6738 logical FAR couplings remain with2753 unique native coefficient owners; no interaction removal, geometry/coefficient approximation or gate relaxation.

Complete process allocator peaks853,474,304B allocated/868,220,928B reserved include construction, every forward/H warmup, measured forward/H, differentiable forward, first current backward, peak-reset gaps and final current/finite checks. Both are below unchanged same-N dense943,718,400B: this selected growing-ka point comparison PASS. Native H2.030842010454122e-15 and current-gradient7.113627934723499e-17 pass original1e-11. Original full-matrix compact retained/allocated/reserved point failures remain preserved; that historical collector omitted warmups/differentiable forward from its per-phase measurements and is not relabelled as complete-interval evidence.

Original collector CLOSED1NEWPASS/XML5977.152s/runner5978.764860432595s/417 stable inputs. Original session22853 drainedrc0; observed runner3119305/pytest3120332 are absent outside the sandbox. Installed Core/native CUDA/GPU1 are recorded. Shared busy GPU timing is not a performance claim. No original all-column reference, solve, producer archive, full regression or full24-case series is run.

Frozen tested baseline89621e7 remains explicit. Current main source applicability is separately documented by continuation proofs:393 other primary modules exact; later operator_error/runtime/lifting changes are unused here and original module inverses are preserved. Never relabel this numerical collector as tested at current HEAD. This selected-point full resource PASS does not qualify all-size scaling/crossover, whole solver or physics. Goal ACTIVE;GSP01/GSP09 OPEN;full scope/gates unchanged;current four-repository final acceptance and at least two independent FINAL audits remain required;production qualification false.

### SHA-bound scaling order admission reuse workflow completed

The current development measurement workflow binds fixed-ka to positive_tail_min_v1 and growing-ka to direct_first_wave_min_v1, the respective closed original3840-edge admissible configurations p11 and p12. Existing failed growing positive-tail admission is preserved. Original two-sphere geometry/materials/q4-singular3/leaf30/separation0.25/G1e-6/H1e-5, all resource/native gates and the complete setup/all-warmups/forward/H/differentiable-forward/cold-current-backward/reset-gap/final-finite interval remain unchanged. No production numerical module changes.

A full series requires a SHA-bound closed-order manifest matching the current complete plan,396-module production source identity and each physical case. Curated original receipt hashes and original producer inputs/tests bind the largest configured order selections. The completed fixed60-edge minimum-p6 native workflow receipt can supply that minimum configuration without re-searching any earlier order. This reuses admissible configured orders, not current minimum-order proof or historical measurements as current repetitions. Every current constructor still checks its native coupling, retained and component budgets. Manual configured constructors report their actual null order_plan; cached selections carry separate provenance and no artificial constructor plan is injected.

Each same-size subsequent repetition consumes the completed current first-repetition selection, source SHA and original matching coordinates/current source/plan/order/native gates. A failed first repetition produces explicit not_launched failures for identical later attempts, without replaying failed setup/admission. Foreign failed first records are rejected. Receipt tampering, changed physical parameters/rules, invalid integer order, changed production/plan identity, unavailable or over-budget kernel bounds, failed/nonfinite/negative native errors and source changes are rejected. Summary validation binds physical parameters and first-repetition provenance; fits use every individual repetition at every declared size and never incomplete tails.

Targeted collector CLOSED39PASS:25NEW order-reuse metadata controls plus14 necessary affected runner/summary controls, XML17.792s/403 stable frozen inputs; no covered native reference/solve/resource PASS is replayed. Original session78252 drainedrc0; runner PID from execution receipt is absent outside the sandbox and short-lived pytest PID was not observed. The initial launcher exited at its missing CUDA_VISIBLE_DEVICES guard before execution/pytest and then received the declared full environment. Actual targeted collector uses installed Core/native CUDA/GPU1. These metadata controls do not establish numerical/physical/scaling qualification.

The24-case full series remains to run at a frozen merged revision:two families×four sizes×three process repetitions. Closed p11/p12 and fixed minimum-p6 selections prevent old planner replay; remaining newly required size/family order searches run once each and current repetitions reuse them. Actual resource repetitions remain separate current measurements. Shared-GPU timing is descriptive only. Original source/gates/failures remain; GSP01/GSP09 OPEN;Goal ACTIVE;four-repository final acceptance and at least two independent FINAL audits remain required;production qualification false.

### Bounded native compact FAR execution and exact archive stage completed

Explicit FMMConfig(far_action_batch=2..32) batches existing original compact electromagnetic translations on CUDA. Each regional action bounds the entire upward/FAR/downward/near sequence to four RHS and each translation to at most sixteen target rows. Order0..12, fixed native complex128, compact or exact-displacement shared compact tables and free-space scalar media are required. Existing default batch1 paths and fourteen archived full payload configurations remain exact; DirectionalFMMConfig is unchanged. New strict cartesian_fmm_batched_compact_config_v1 binds every configuration field. Periodic, complex64, unsupported order/storage and invalid structural batches reject before finite scans/stacks.

Original scalar Green/curl/charge formulas use batched matmul without new kernels, coefficient polynomials or near quadrature. Native reversal parity and every logical FAR interaction remain; per-pair accumulation keeps original order. All stacks/tables are transient; retained ownership/capacity is unchanged. Existing fixed-operator autograd forwards save no generated tensor and route first/higher current derivatives through the original native Hermitian action. A conservative visible-tensor preflight at maximum order12/pairs32/RHS4/rows16 is121,388,800B below128MiB; this is not a measured allocator/library or whole-system resource certificate.

Fourteen distinct Maxwell checks PASS: eleven NEW and three necessary affected old scalar/FFT controls. Original v2 remains10PASS/3fixtureERROR, source418 stable. The blocked positive controls continued alone in v3, obtaining3PASS with complete120-column original weak matrix error3.9795055549581184e-16, native Hermitian dot2.225837265142875e-16,15 noncontiguous RHS, first/higher current gradients, zero saved forward tensors, exact1,013,696B retained capacity, native reversal and mutation refusal. A single full original weak matrix supplies all new physical controls and the public all120-row current residual1.3686043304280155e-13 at the unchanged1e-10/atol0 gate. Public native solve/strict exact restore PASS; one result is saved once. This dense consistency/residual evidence does not certify continuum physics or quadrature accuracy.

New p12/32pair/4RHS boundary initially failed before coefficients: v3 complex-eps public material compilation TypeError, then v4 incorrect sigma keyword TypeError. Both original failures and whole test sources remain preserved. Only the failed new fixture changed to declared sigma_e conductivity with real eps2/mu1.3; all396 primary sources and all successful controls remain exact. The failed-only v5 now1PASS, original forward1.5139264402408674e-17/H2.1592377384656235e-17 and native dot4.37790243810346e-16 below unchanged1e-11. Two original coefficient owners and their reversals exercise all32pairs without duplicate polynomial construction. The complex-eps compiler limitation remains unresolved and no complex-eps qualification is inferred.

One NEW Bench check PASS with69 production modules unchanged. Its independent Scene/full config is declared before loading terminal v3 successful public artifact b24923a12fb3ee40b7628f1b9cba120eece225746f7d9580c84ac7ee19f1b4ea. Constructor/solve tripwires, exact schema11 J/M bytes, strict version restoration and batch/table/near/kernel/shift manifest/native tamper rejection pass. Unqualified-candidate role and false qualification remain. Original v3 aggregate3PASS1FAIL is preserved; its successful public subtest is selected explicitly. No repeated Maxwell solve, original matrix, order search or producer archive.

All native checks use GPU1 outside the sandbox with installed Core. Original sessions68023/v2,49336/v3,86096/v4,77182/v5 and17428/Bench are drained; observed v3 runner589002/pytest589162 are absent, other short-lived pytest PIDs were not observed and remain null. Complete receipts/XML/hash closures and source inverses accompany the assessment. No per-edit full regression or covered PASS replay. This GSP09/GSP10/GSP11 implementation stage leaves resource/complexity/physics and all final gates open. The original24-case all-point series remains LIVE and untouched at frozen1204329; changed scalar numerical source in this stage prevents relabeling that earlier-source series as current optimized evidence. Shared-GPU timing is descriptive. Original resonant f30 retained-capacity rejection, all GSP01 failures and all other open capabilities remain. Goal ACTIVE; complete scope/thresholds and current four-repository final acceptance with at least two independent FINAL audits remain required; production qualification false.

### Linear complex scalar material admission correction completed

The original compact FAR stage exposed a linear Material(eps_r=2+0.05j,mu_r=1.3+0.02j) compiler TypeError: tpa_sigma_scale always attempted a real sqrt(eps_r), even with no TwoPhotonAbsorption declaration. The primary Material property now returns exact0 for no absorption declarations before any real-index conversion. All395 other primary modules remain exact; actual TPA formulas and their existing host-index restrictions remain. This corrects the existing finite-frequency scalar material contract without dropping imaginary constitutive values or loosening passivity/background gates. General complex DC/TPA/nonlinear qualification remains open.

Targeted GPU1/installed-Core collector CLOSED11PASS/XML8.597s/409 stable inputs: nine NEW host/public controls, one original failed maximum compact boundary under the exact original complex eps/mu context, and one necessary affected pre-existing real TPA formula control. New host controls forbid index sampling when no absorption exists, preserve passive outgoing branches and base metadata, count electric/magnetic conduction once, and retain gain/lossy-background rejection. Active complex TPA still refuses its unsupported real-index interpretation. Only original failed boundary context is retried; prior conductivity-boundary PASS and original120-row/public solve/archive/planner are not replayed. Original v3 TypeError and original producer aggregate failures remain archived.

New public Scene/Simulation preparation with complex eps2+0.05j/mu1.3+0.02j at90MHz/subdivision0/q4-singular3/native compact-shared reciprocal/separable p6/batch32 passes one complete original60-current matrix2.9926092838936833e-16, original H4.536429576002527e-16 and current-gradient3.852532514722163e-17 at unchanged1e-8 matrix/H and1e-11 gradient gates. Imaginary eps/mu are retained exactly. The original p12/32pair/4RHS boundary now forward1.405685772857564e-17/H2.2757372762507546e-17/native dot7.266535458506695e-16 below original1e-11. There is no solve, Result save, Bench archive or duplicate numerical producer. Host configuration/oracles introduce no CPU field/action fallback.

Original session43156 drainedrc0; runner604720 from execution receipt is absent outside the sandbox; short-lived pytest PID was not observed and remains null. Independent PHASE audit precedes human commit. This stage grants only implementation/selected discrete consistency evidence, no continuum physics/quadrature, complete allocator/resources/scaling/populations or production qualification. Original all-point24 series remains LIVE at frozen1204329 with no provider changes or relabeling; all other LIVE jobs and original GSP01/f30/open capability failures remain. Goal ACTIVE;GSP01/GSP09 OPEN; original complete scope/thresholds, current four-repository final acceptance and at least two independent FINAL audits remain required.

### Current bounded compact FAR original large resource intervals completed

Two NEW independent process intervals measure the current bounded FAR32 execution at baseline6e7cf36 on the original3840-edge physical configurations. All396 primary numerical modules remain exact. Closed original p11 fixed-ka and p12 growing-ka admissions bind the same physical mesh/frequency/quadrature/leaf30/separation.25/native G1e-6/H1e-5 and unchanged2GiB action/1GiB component gates; original planner, reference matrix, public solves and archives are not replayed. The execution interval starts before fresh geometry/construction and includes both warmups, forward/H, differentiable forward, cold first current backward, allocator reset gaps and final finite checks. Every original6738 logical FAR pair and both RHS/seed1729 remain. The original owned categories are checked exactly, and actual bounded compact calls enforce32 pairs/two RHS/reverse owners.

fixed_ka: owned585948608B, complete allocated783855616B/reserved845152256B versus unchanged same-N dense943718400B; allocated-below=True, reserved-below=True. Native H=2.6272818859978878e-15, current derivative=7.550143747730608e-17 under strict1e-11/finite gates; actual native gate=True. Targeted pytestPASS XML3010.652s;425 frozen sources stable. Any failure is retained as a failure, with the complete raw receipt written before assertions.

growing_ka: owned668235432B, complete allocated895041536B/reserved985661440B versus unchanged same-N dense943718400B; allocated-below=True, reserved-below=False. Native H=1.4310617894142624e-15, current derivative=7.371883846306649e-17 under strict1e-11/finite gates; actual native gate=True. Targeted pytestFAIL XML3054.749s;425 frozen sources stable. Any failure is retained as a failure, with the complete raw receipt written before assertions.

Both original handles are drained and observed runner/pytest PIDs absent outside the sandbox. Independent PHASE review precedes the human evidence commit. This is selected-point whole setup/action/current-backward measurement, not all-N/crossover/whole solver/host/teardown or physics qualification. Shared GPU timing is descriptive only. The separate original24-case series remains frozen at1204329, not relabelled as current optimized source. Old f24/f30 capacity and other resource/physics failures remain. GSP01/GSP09OPEN; Goal ACTIVE; final scope/gates, current four-repository acceptance and at least two independent FINAL audits remain; production qualification false.

### Cartesian FMM component capacity refusal before numerical setup completed

The primary scalar free-space Cartesian system constructor now applies all
table/near/projection component budgets to the original structural capacity
categories immediately after total retained admission. Previously only the
shared-compact table component had this early guard; other table modes, near
and projection could enter numerical moment setup before cumulative refusal.
The new guards reuse the one existing plan, sum the original full projection
categories (centers/radii, leaf maps, vector/divergence moments and shifts), and
retain total-rejection priority and later constructed-owned/cumulative checks.
They remain after any optional kernel-order search, not ahead of all planning.
No extra geometric/planner traversal, altered allocation or kernel/coefficient/
quadrature/action/derivative/configuration identity is introduced. An oversized
component stays oversized; this does not make the original f30 problem fit.

Thirteen DISTINCT GPU1/installed-Core targeted controls PASS: eleven NEW
refusals, one necessary affected existing shared-table guard and one necessary
affected aggregate receipt/priority guard. Actual two-body120-current CUDA
support geometry and original structural plans cover table/near/projection
refusals across dense/compact/shared-displacement/FFT representations. Selected
component budget1B is checked while total capacity is admitted; callbacks forbid
numerical moments and original weak integration. The aggregate control retains
the existing MultipoleCapacityError receipt and priority with all budgets1B.
Each constructor traverses the capacity planner once. No old successful
operator/matrix/quadrature/solve/producer/archive/resource check is replayed.

Original session89999 drainedrc0; runner706757 from the execution receipt is
absent outside the sandbox; short-lived pytest PID was not observed and remains
null. All405 frozen inputs stable. XML4.637s, runner5.685548301786184s.
Only fmm_system.py changes;395 other primary modules and the preserved original
module inverse remain exact. Independent PHASE audit precedes human commit.

This stage improves actual pre-allocation refusal behavior, without resource/
physics or production qualification. Original f30 dyadic-capacity and growing
row8 resource collectors remain independently frozen/LIVE; original growing
FAR32/row16 reserved985661440B exceeds unchanged943718400B and remains failed.
The original all-point series remains at1204329, not relabelled as current.
GSP01/GSP09OPEN; Goal ACTIVE; complete original scope/gates, current four-repo
final acceptance and at least two independent FINAL audits remain required.

### Original growing compact FAR row8 resource successor completed

The original failed growing-ka FAR32/row16 point has one failed-only successor
with the existing m2l_row_batch=8 configuration. All396 tested primary numerical
modules remain byteexact at f11d8d1. The whole original test is preserved; its
only source delta inserts that configuration. Original3840edges/7680currents,
2.4GHz/p12/direct-first-wave admission, two spheres, leaf30/separation.25,
quadrature4/singular3, all6738 logical FAR pairs, two RHS/seed1729 and exact
668235432B retained categories remain unchanged. No original kernel planner,
matrix, passed fixed-ka resource point, public solve or archive is replayed.

One NEW GPU1/installed-Core targeted resource control PASS. The entire fresh
setup through warmups, F/H, differentiable forward, first-current backward,
reset gaps and final finite checks measures allocated871140352B and
reserved931135488B, BOTH strictly below unchanged943718400B. Native H relative
error3.640701694855753e-15 and current-gradient relative error
7.814810111862312e-17 pass the original1e-11 gate. Actual
owned categories and all logical pairs match original admission. No empty_cache,
allocator setting change, relaxed gate, synthetic workspace deletion, smaller
mesh or changed retained budget. Timing on the shared GPU is descriptive only.

Original row16 reserved985661440B failure remains preserved; this successor
does not revise its outcome. The successful fixed-ka point is not rerun and
is not a row8 measurement. Original session77195 drainedrc0; observed runner
681415 and pytest681641 are absent outside the sandbox, all439 frozen sources
stable, XML2816.550s, runner2817.781495344825s.

Incoming main cf80e54 differs from this numeric snapshot only in the separately
audited host-only component-refusal block. Exact one-block inverse and395 other
primary hashes prove the source boundary; original actual table181808120B,
near259874344B and projection226308368B pass its unchanged1GiB budgets.
The measured collector retains f11d8d1 identity. This evidence/test/docs stage
does not edit main numerical modules or claim a fresh cf80 numerical run.

This is a selected setup/action/first-backward allocator point, excluding solve,
teardown, host peak and full solver state. All-N/crossover/physics/production
qualification remains false. LIVE dyadic f30, all-point, tensor and magnetic
providers remain immutable. GSP01/GSP09OPEN; Goal ACTIVE; original final scope,
thresholds, current four-repository acceptance and at least two independent
FINAL Goal audits remain required. Independent PHASE audit precedes human
commit, and root/checkpoint append once only after merge.

### Exact native displacement catalog bounded batches completed

The existing primary IEEE equal/opposite displacement catalog increases ONLY
its bounded native staging chunk32 to1024. Native float64 subtraction,
finite/nonzero guards, sign/zero canonicalization, exact bit keys, first-owner
order and every logical reverse binding stay unchanged. Index uploads and
IEEE/orientation record downloads are bounded per1024logical pairs rather
than32. No approximate displacement bins, CPU numerical geometry, owner/pair
removal, persistent cache or changed coefficient/near/projection/order/action/
configuration implementation. All395 other primary modules stay byteexact.

Seven DISTINCT targeted GPU1/installed-Core controls PASS: one NEW ordered
nonbinary/nextafter/signed-zero input of14813logical pairs with
duplicates/reversals across boundaries, full exact owner indices and all
bindings equal to the preserved original32 module, actual1024chunk bound and
empty-catalog identity; six necessary affected invalid-input refusals retain
zero/nonfinite/precision/trainability/index restrictions. Unique owners
2013; actual chunks15 versus original
463. These are actual new catalog inputs, not a replay
of any old physical geometry/planner/reference/operator/solve/archive/resource
PASS. Original native comparison0.9284621328115463s and candidate
0.06031939387321472s are shared-GPU descriptive timings, with no speedup
or all-N qualification. Numerical staging remains CUDA; only integer/IEEE
metadata is processed on the host. No numerical fallback is added.

Original session62267 drainedrc0; execution runner
738416 absent outside the sandbox; pytest PID observation
None is retained honestly. All405 frozen inputs stable,
XML4.177s, runner5.2356750797480345s. Exact original module and
inverse patch remain preserved. Independent PHASE audit precedes human commit.

This change can alter temporary setup workspace and allocator behavior.
Existing row8 resource measurements remain frozen f11d8d1, prior numerical
snapshots are not relabelled current, and no new whole-resource result is
claimed. Original row16 failure and live dyadic f30/all-point/tensor/magnetic
providers remain unchanged. All-N/crossover, full solver/setup/action/backward/
host resources, continuum physics and production qualification remain open.
Goal ACTIVE;GSP01/GSP09OPEN; original final scope/gates, current four-repository
acceptance and at least two independent FINAL Goal audits remain required.

### Original tensor material plane-tail diagnostics completed

Two NEW GPU1/installed-Core instrumentation controls inspect the actual binary
epsilon=2*diag(1.25,1,.75), mu=1.3*that metric, original0/.37 orientations,
without constitutive projection, hidden loss, relaxed SPD admission or any
production edit. Independent MP100 reconstructs the actual Maxwell6x6 symbol,
longitudinal Schur elimination, radiation eigenspaces and physical-source
residues. Twelve NEW complete-plane points100..1e8 all pass the unchanged.005
selected-point comparison; largest balanced relative error5.808419816815235e-5.
This does not qualify regular remainder, spatial quadrature, covariance or HB.

Six NEW regular-plane points use three wavevectors NOT in the closed complete
diagnostic: (1000,1000),(500000,700000),(100000000,100000000), both original
orientations, positive z=1/(k0*|q|). MP100 static Coulomb upper-half-plane roots,
Hessian residues and electric/magnetic physical scales remain100digits until
AFTER full-minus-static subtraction. Native unchanged compiled full and static
planes are subtracted as in the production spatial integrator and compared
separately in EE/EH/HE/HH. Raw selected outcome: three of six points fail,
eleven of24blocks fail unchanged.005, despite instrumentation pytestPASS.
Rotated(500000,700000) EE/HH relative errors3219618.012463433 and
2494769.1897350643; complete balanced error1.9227730825087697e-5. Unrotated
(100000000,100000000) all four regular blocks fail; rotated high point also
has complete spectrum error.3330760542394404 and regular failures up to
3.9855729791721485e15. These are real native capability failures, never
converted to physical or production qualification by an instrumentation PASS.

Saved exact diagnostic tensors are reused ONCE for GPU-only component analysis,
without re-evaluating modes or MP references. Even ideal MP100 complete/static
results rounded separately to double and subtracted lose diagonal remainder
accuracy: at highq unrotated EE/HH errors.492371907/.833548949; rotated
.571814672/1.136189660. Thus eigensolver changes alone cannot recover these
small terms: a stable factored dynamic-minus-static construction is required.
This identifies a mechanism to address before another expensive spatial run;
the exact old failing displacement is unavailable and no claim is made that
these standalone nodes reproduce the precise old20h failure query.

Original covariance-v4 remains CLOSED FAIL before orientation0 HB preparation.
The inspected dyadic module adds configuration/dispatch class fields; the
matched-SPD module adds a closed-form helper. Every original top-level function
in those modules remains exact in AST, and spectral_linalg remains byteexact. Original
whole-snapshot identity is preserved, not relabelled current. First full-plane
prelaunch NaN serialization P2 and regular prelaunch finite-norm P2 were fixed
BEFORE numeric launch, preserving whole original tests, original inputs,
rejection reports and exact inverse proofs. No completed numerical check was
replayed. Nonfinite/zero reference metrics reject, never pass.

Original full session24258 and regular11828 drainedrc0; full runner known from
execution absent, unobserved pytest PID honestly null; regular observed764897/
765040 absent outside sandbox. All412/417 frozen inputs stable. XML full
4.59s, regular5.018s.
Two instrumentation PASS reflect completion only. All396 production modules
remain unchanged; no original spatial quadrature, original weak matrix, HB
solve, producer/archive, resources or covered successful point test is replayed.
LIVE f30, all-point, tensor and magnetic providers remain immutable.

Independent PHASE audit precedes human commit. GSP01/GSP05/GSP06/GSP09 remain
OPEN; Goal ACTIVE; complete original scope/gates, current four-repository
final acceptance and at least two independent FINAL Goal audits remain required.
General tensor regular tails/spatial/covariance/HB and production unqualified.

### Factored evanescent tensor plane remainder primitive completed

One native CUDA complex128 internal primitive in the primary dyadic Green
module constructs regular evanescent plane blocks with explicit inverse-length
polynomial factors. It preserves actual fixed real symmetric SPD epsilon/mu,
including genuinely nonproportional and noncommuting coefficients; no material
projection, symmetrization, removed imaginary response, SPD tolerance weakening,
artificial damping, finite spectral cutoff or CPU production fallback.
Original17 top-level functions and all395 other production modules remain exact.
The primitive is NOT wired into spatial integration, public solvers or persistence.

The transverse-aligned tangential similarity S=diag(s,1,s,1) balances the
Maxwell generator as A=A0+x*A1+x^2*A2, x=1/s. Embedding and forcing have
explicit small polynomial corrections. Coupled Newton matrix-sign iterations
retain projector differences through the inverse difference identity. A bounded
outgoing generator removes incoming exponential growth; an upper triangular
matrix exponential retains the propagator difference. EE/HH subtract their
Coulomb baseline through the three-term product difference. EH/HE retain full
physical curl terms; normalized x=0 cross terms can survive unbalancing and
must not be subtracted with Coulomb. Singular/nonconverged/nonfinite/unresolved
projectors and unsupported domains refuse before returning a spectrum.

The original three FAILED plane points reuse closed exact MP100 arrays without
recomputing reference/native full spectra, older twelve complete-plane controls,
three successful regular points or expensive original spatial covariance/HB.
First numerical candidate CLOSED1FAIL/XML4.307s:
six EE/HH blocks passed1.7e-14..2.0e-13, six EH/HE failed.7045..7058. The failure
and original full primary/test source remain preserved. Original prelaunch
review incorrectly treated the normalized baseline as pure diagonal Coulomb;
that report is retained and explicitly corrected by the incremental audit.

Terminal-only correction inserts exactly one electric/magnetic channel mask:
EE/HH select their unchanged factored difference; EH/HE select full propagation.
An exact source inverse and channel-preserving continuation proof bind the
closed six diagonal PASS without rerunning them or relabelling the old snapshot.
Only six FAILED cross blocks rerun, now PASS5.3e-16..5.8e-15 unchanged.005 gates,
XML3.997s. One genuinely NEW full noncommuting SPD
medium at two NEW q vectors is compared against independent full6x6/Coulomb
MP100 subtraction; all eight selected blocks pass unchanged.005. Four NEW
physical-domain checks reject1e-20 imaginary loss,1e-16 roundoff asymmetry,
indefinite coefficients and material gradients without modifying actual input.
General collector5PASS/XML4.375s. Saved arrays
support subsequent reuse. No previously covered numerical PASS is replayed.

Original sessions26462/42965 and the general session are drained; execution-known
runner identities absent outside sandbox, unobserved pytest identities honestly
null. Frozen closure415/429/411 inputs; GPU1/installed Core. Reference artifact
path/SHA pairing is explicit; absence of both skips ordinary repository tests,
partial/missing/mismatched input refuses. Original prelaunch test-contract P2,
whole original test and exact inverse retained; no prelaunch numeric replay.

This closes selected internal spectral comparisons only. Spatial quadrature,
original20h covariance failure, tensor cell/public AC/HB and general complex,
trainable/indefinite tensor response remain OPEN/unqualified. No original matrix,
solve, producer/archive, capacity/resource or full regression replay. LIVE f30,
all-point scaling, magnetic DAG and tensor providers remain immutable. Independent
PHASE audit precedes human-owned commit. GSP01/GSP05/GSP06/GSP09OPEN; Goal ACTIVE;
original scope and all acceptance thresholds unchanged. Current four-repository
full acceptance and at least two independent FINAL Goal audits remain required;
no production qualification.

### Factored real-SPD spatial tail integration completed

The primary dyadic Green module now exposes the explicit integration rule
factored_spd_tail_v1 for fixed exact-real symmetric SPD free-space media.
Distance-independent Maxwell/sign/projector state is compiled once for each
quadrature grid. All original lower-wavevector modal nodes and every original
infinite-tail node/weight remain; the strict evanescent bound selects the
factored polynomial remainder. EE/HH retain their explicit Coulomb differences;
EH/HE retain full physical curl terms. Existing native_v1 and closed-form rule
defaults and old default payload/key omission remain. No constitutive projection,
removed imaginary response, symmetrization, relaxed SPD gate, artificial physical
loss, finite transverse cutoff or CPU production fallback is introduced.

An exact incoming-sector generator reassignment avoids roundoff leakage from
an unused zero eigenvalue sector: exp(t*(i*A*P-alpha*(I-P)))*P=exp(i*t*A)*P
for commuting exact projectors. Its explicit difference includes +t*alpha*dP.
The physical outgoing eigenvalues are unchanged. This algebraic identity does
not certify rounded projectors or every spatial point. Native propagation uses
bounded256-node batches; construction workspace is not independently qualified.

Compiled grids/cache entries own and guard retained tensors, including identity,
version and requires_grad. The latter can change without a tensor version bump.
Unique GPU storage inventory includes all compiled tail state; cache reuse does
not reconstruct modal/sign matrices. Unsupported lossy/asymmetric/indefinite/
trainable samples, paired regional contrasts and periodic composition reject
before the new numerical strategy. Runtime, contrast and periodic primary
modules contain only the corresponding early guards;392 other primary modules
remain byteexact to baseline. General complex, contrast, periodic, material
gradients and public tensor/HB response remain required future Goal work.

Prelaunch review found one undefined-device P1 and five P2 gaps in gradient-mode
binding, per-query gates, failure evidence, unsupported-domain coverage and query
batch validation. Whole original primary/test/input and original audit remain
preserved. The exact correction inverse binds the prelaunch source; no first
candidate numerical run occurred. Clean incremental prelaunch review preceded
the original numerical launch. Historical audit input hashes refer to their
preserved preaudit snapshots, not to the later final417-input launch manifest.

Original session64606 CLOSED0, XML5.867s,3 targeted PASS,
all417 frozen SHA stable, execution-known runner1396371 absent outside sandbox.
The short-lived pytest PID was not observed and remains honestly null. GPU1,
installed Core, native CUDA production. Two genuinely new spatial points for
epsilon=mu=diag(2.4,1.7,1.2),30MHz have independent MP100 matched-SPD analytic
references; all eight individual query/block .005 gates pass, maximum relative
error6.287720281342211e-12. Adaptive independent
radial/azimuthal checks converge at orders[64, 32] with
unchanged rtol2e-4/atol1e-10. This is a successive-order estimator, not a formal
error bound. Actual unique retained cache35207488B
is below268435456B; it excludes caller/workspace/allocator/whole solver peaks.
One cache-hit control uses saved values with exact equality and forbids new
compilation. Gradient-flag mutation, in-place mutation and invalid query-batch
controls reject. New payload/unsupported runtime-periodic/contrast/periodic
kernel controls pass; the necessary affected native cache protocol uses synthetic
CUDA metadata without eigensolve/reference replay. Native/reference arrays are
saved for future reuse. No previously covered plane/spatial/HB/matrix/solve/
producer/archive/capacity/resource/full-regression PASS is replayed.

The original public tensor AC task also independently CLOSED1FAIL after
122779.59s: outgoing mode split unresolved at a grazing or defective spectral
point during cell-source integration. Its original session99995 is drained,
720 frozen SHA stable and both observed PIDs absent; it has no execution/Core
receipt and no identity is invented. Original f30 dyadic-center preflight closed
with its instrumented test PASS but actual capacity NOT admitted: planned
27026890736B, table23755200000B, near2974652336B exceed unchanged2GiB/1GiB gates.
Neither old collector used this new spatial rule or is relabelled current-source
evidence. Both failures remain preserved. Still-running providers are immutable.

This stage closes selected spatial integration and fixed-cache controls only.
Original covariance, public tensor AC/HB, full production physics/resources,
populations and current four-repository acceptance remain OPEN/unqualified.
GSP01 stays OPEN. Goal ACTIVE; original scope and every acceptance threshold
remain unchanged. Independent PHASE review precedes human-owned commit and one
root/checkpoint append after merge. At least two independent FINAL Goal audits
and full current acceptance remain required; no production qualification.

### Public exact native dyadic expansion centers and original f30 capacity stage completed

The primary Morton support tree optionally owns fixed dyadic-cell expansion centers. Complete original support boxes and original support-midpoint default topology/payload remain; the new explicit-center sphere encloses every complete box corner. Native center projections own separate exact-size storage, and mutation/version/trainability guards bind all consumers. The same centers/radii reach complete ordered-pair partition, current projections, kernel order admission, exact IEEE equal/opposite displacement sharing and structural retained inventory. No approximate displacement bins or logical pair removal. Existing directional and periodic consumers reject this unsupported center geometry before native projection. One outward radius ULP is not a formal floating-point interval certificate.

Public FMMConfig(expansion_center_policy="dyadic_cell_v1") has strict complete-field cartesian_fmm_dyadic_centers_config_v1 identity and free-space/float64 compact order0..12 limits. Defaults omit the new field from old identities. Public Scene/Simulation constructs separate exterior and each-body trees, binds policy in current guards and metadata, and supports exact physical-lifted Result recovery without reconstructing or solving the numerical operator. Bench declares the same full physical/config identity independently before loading the one new completed producer, retains exact J/M bytes/native schema11 and rejects changed numerical identities. All69 Bench numerical modules remain unchanged.

Nineteen DISTINCT targeted controls PASS:11NEW internal geometry,7NEW public/config controls (including necessary affected default payload identities),1NEW Bench archive. No previously passed original matrix/solve/producer/archive/order/resource check is replayed. New two-body90MHz/subdivision0 complete120-current original weak reference is built once and reused for all columns,15 noncontiguous RHS/native H, first/higher fixed-current gradients, one actual solve and all-row original residual. Actual fresh-center order p6 is rejected; p7 admitted under originalG1e-6/gradient1e-5 gates. Complete column error2.145507121824446e-12, native H7.798434643118549e-15, current derivative5.829051385822409e-17, complete original residual1.4201599056689806e-11 below1e-10/atol0. Saved forward tensors0; actual/planned owned1474264B. Exactly one new producerSHA55473527104cc4967d4fa7edb9f94c81f5134e471e052e6bf1536689c1e87b8f is saved; exactly one Bench consumer archives it without fresh solve.

Original f30 geometry remains27000edges/54000currents/18000faces,r=.1m,414635862.3319615Hz,epsilon12+.01j; only centers and leaf8 change from old failed leaf30 midpoint compression. Two independently owned physical trees counted; .005 physical and1e-10/atol0 original residual goals,2GiB action and1GiB table/near/projection budgets unchanged. Minimum p6 planned27026890736B, table23755200000B, near2974652336B, projection292271360B; gates{"action": false, "near": false, "planned_projection_bytes": 292271360, "projection": true, "table": false}. Actual configured-capacity admission=False, reason=minimum_order_retained_or_component_capacity_exceeded, selected_order=None. Kernel order search performed=False; only actual minimum-admitted geometry is searched. One NEW instrumentation PASS grants no capacity admission unless its raw gates pass and grants no physical/resource/production qualification. Original f24 field and old f30 capacity failures remain preserved. No dense46,656,000,000B global matrix, expansion, original quadrature, compressed action, solve or physical field is constructed by this preflight.

A separate source-bound host integer inventory of the actual two12197-node trees shows the complete native projection component fits only throughp10 (991120064B); p11 needs1261040000B and p12 needs1575981536B, both above unchanged1073741824B. This exact retained-category analysis performs no projection/field/action or repeated geometry/kernel computation. It does not establish native kernel admission; the actual configured outcome remains the raw preflight result above.

Original geometry/public/Bench/f30 handles are drained, recorded observed PIDs absent (short Bench pytest PID remains honestly null), frozen sources stable. Independent PHASE audit precedes separate human Maxwell/Bench commits and append-only root/checkpoint after merge. This stage implements selected geometry/public/archive capabilities; it does not qualify continuum quadrature/physics, automatic magnetic changing-carrier, tensor/nonlinear/periodic FMM, whole solver/setup/action/backward/host resources, all-N/crossover/populations or current four-repository final acceptance. Existing growing FAR32/row16 reserved985661440B failure remains, and its separate row8 successor stays independently bound. The original all-point series remains at1204329 with no relabeling. Goal ACTIVE;GSP01/GSP09OPEN; complete original scope/gates and at least two independent FINAL Goal audits remain; production qualification false.

After original f30 source/session closure, the provider adopts the exact separately audited cf80e54 component-refusal block. Its whole original tested fmm_system module remains preserved; one-block inverse restores that exact module. The following separately audited catalog and explicit spatial-strategy adoptions have separate full-module continuation proofs. Old public actual table/near/projection categories pass every unchanged budget, so the extra host-only guard rejects no admitted public construction and changes no numerical path. The old numerical collectors retain their own frozen snapshots; combined current source is not presented as a fresh combined numerical run. The original13 component-refusal controls are independently preserved rather than replayed.

The post-terminal merge also preserves incoming aab1ebd bounded native displacement catalog staging: chunk32 to1024, exact subtraction/sign/zero/IEEE keys and original first-owner/all-binding identities unchanged. The whole original tested fmm_storage module is separately preserved; exact three-substring inverse restores it. Seven independently audited catalog controls remain bound to their own snapshot and are not replayed. Temporary setup workspace and allocator behavior can differ, so old public/f30/resource measurements are not relabelled as current combined source or peak qualification. Original eleven dyadic implementation modules plus the adopted catalog and three new spatial integration/contrast/periodic modules constitute fifteen current primary differences from the tested baseline;381 others remain byteexact. The fourth spatial-stage module, runtime, also retains its original center-policy change and adopts only the independently audited two-line unsupported periodic-rule guard.

After original closure the isolated provider adopts the separately audited factored real-SPD spatial stage from its exact completed manifest/merge receipt. Whole original tested dyadic_green, dyadic_contrast, periodic_dyadic and runtime modules are preserved. For contrast/periodic and the combined runtime, exact two-line guard removal restores the tested original modules; runtime also retains its center-policy implementation. The original f3 added-primitive inverse restores the old dyadic module, while the current spatial changes remain bound to their separate phase audit. The new explicit rule is wired only when requested; old scalar public/archive/capacity paths do not use tensor spatial compilation, the new guard predicates are false on those paths, and old default payload/key omission remains. Three separate spatial controls/eight MP100 block gates are preserved at their own current snapshot without replay. This is source applicability, not a fresh combined tensor/numerical/resource qualification; old collectors are not relabelled.

### Two-factor paired evanescent tensor plane remainder completed

One new internal CUDA complex128 paired plane primitive preserves actual fixed
exact-real symmetric SPD epsilon/mu samples, including nonproportional and
noncommuting pairs and near-zero constitutive differences. The primary dyadic
coefficient and field/source factor bodies are extracted byteexact into shared
helpers. Two full-module inverse proofs restore the original single-medium
module and retained state; no old successful spectral/spatial control is replayed.
The contrast module contains only new helpers/factory/class; deleting that
segment restores its entire original module.394 other primary modules are exact.

Actual constitutive differences are formed before rotation congruences and
inverse Schur reductions. Li-Le=-Li*dMll*Le and explicit product differences
retain the material factor in every polynomial coefficient, field embedding
and source forcing. Inverse transverse length x remains explicit independently.
An upper triangular four-state16x16 matrix encodes inside/full, inside/static,
outside/full and outside/static samples, dynamic differences, material
differences and their mixed difference. Newton sign iterations carry these
off-diagonal factors without subtracting rounded full projectors. The same
four-state homomorphism applies to rectangular field/source factors and the
propagation exponential. EE/HH select the mixed material/dynamic03 block;
EH/HE select the complete physical material02 block, preserving curl terms.

One shared auxiliary incoming-sector scale across all four states preserves
exp(t*(i*A*P-alpha*(I-P)))*P=exp(i*t*A)*P exactly for commuting projectors.
Physical outgoing eigenvalues and actual material bits remain unchanged. No
material projection/symmetrization/removed imaginary response, weakened SPD
gate, physical damping, finite cutoff or CPU numerical production fallback.
Diagonal sign residual is NOT a relative accuracy certificate for tiny mixed
differences; selected independent comparisons establish only their tested scope.

Both media require common frequency/device/precision and transverse coordinates
beyond BOTH strict evanescent bounds. Compiled state owns its retained tensors
and guards dictionary keys, tensor identity/version/requires_grad and both
constitutive samples.16x16 propagation is batched64 nodes; full grid construction,
caller/output storage and whole allocator/solver/host peaks remain unqualified.
The paired cache inventories actual unique GPU storages. This primitive is NOT
wired into spatial quadrature, public tensor AC/HB, periodic or persistence paths.
Their explicit unsupported paired spatial guard remains. General complex,
trainable/indefinite materials and periodic/full public scope remain required
future Goal work; no narrowed success definition or production qualification.

Original session26552 CLOSED0,7 targeted PASS/XML4.825s,
all414 frozen inputs stable, execution-known runner1966224 absent outside sandbox;
unobserved short-lived pytest PID honestly null. GPU1/installed Core. Four
genuinely NEW MP100 references resolve full6x6 Maxwell and independent Coulomb
plane spectra for actual finite noncommuting pairs and binary perturbations
2^-40/2^-50. Each regular remainder AND the material difference remain MP100
before double conversion; the new explicit helper return_mp option preserves
the old default helper body/contract through an exact source inverse. Reference
arrays are saved before native compilation, native arrays saved once afterward.
All16 independent EE/EH/HE/HH .005 gates pass, maximum relative error
3.157632351882224e-13. No proxy medium or
separately rounded double spectrum subtraction is used in the reference.

One NEW129-node exact-zero pair verifies all36 spectrum entries and two-point
integral exact zero; actual exponent batches64/64/1. One cache hit forbids new
compilation; unique retained2726832B below16MiB.
Gradient-flag mutation without version bump and version-only mutation both
release/refuse before action. Two pair-specific early negatives verify common
frequency and the maximum exterior/interior evanescent bound before polynomial
work. One independent static prelaunch review preceded the only numerical run;
its original411-input SHA binds preserved preaudit-input-v1, while the final414
launch adds the audit/copy SHA. No original successful plane/spatial/HB/reference/
matrix/solve/producer/archive/planner/resource/full regression is replayed.

Separately, original tensor scalar-limit cell response now has terminal XML
1FAIL/405440.076s: volume source matrix outer_order_budget_exhausted. Original
observed PID864162 is absent and session70271 is missing. Its return code remains
honestly unknown and the handle was NOT drained; no restart or acceptance is
invented. All831 actual before/current SHA match across its Maxwell/Bench
providers. Original receipt remains unmodified with stale running/head metadata;
the separate terminal-observation receipt records the gap and actual source
identity. It has no execution/Core receipt. This failure, public tensor AC grazing
failure, original covariance and f30 capacity failure remain OPEN/unqualified;
none ran this new primitive or is relabelled current-source evidence. Remaining
original all-point scaling and magnetic DAG jobs remain LIVE and immutable.

This closes selected internal paired plane precision/ownership only. Paired
spatial convergence and low-wavevector outgoing-sheet resolution are separate
next dependencies before public tensor AC/HB recovery. GSP01 stays OPEN; Goal
ACTIVE. Original complete scope and every acceptance threshold remain unchanged.
Independent PHASE audit precedes human-owned commit and root/checkpoint appended
once after merge. Complete current four-repository full physics/complexity/
resource/population acceptance and at least two independent FINAL Goal audits
remain required; production qualification false.

### Explicit two-factor paired real-SPD spatial integration completed

The new explicit spatial_rule=factored_spd_pair_tail_v1 and radial_map=
paired_spd_reference_branch_v1 connect the audited two-factor paired plane
remainder to free-space spatial quadrature. Actual fixed exact-real symmetric
SPD samples remain unchanged. A common grid retains all contributing low modal
minus-static nodes and all nodes beyond BOTH strict evanescent bounds, where
the factored paired remainder is used. The old single-medium strategy retains
its paired refusal; native defaults and their omitted payload fields remain.
The new rule is also single-medium capable through the existing compiled grid.
Periodic public and regular-kernel composition refuse this free-space strategy
before numerical work. No constitutive projection, imaginary loss removal,
symmetry/SPD weakening, outgoing damping, finite cutoff or CPU production.

Three radial intervals cover the whole transverse half-line for each angle:
below the smaller reference ellipse, between ordered ellipses, and above the
larger ellipse to infinity. Both interior endpoint square roots have explicit
rho*d rho Jacobians. Equal reference branches have zero middle weights; their
zero-measure placeholders are below the branch, avoiding evaluation at poles.
Reference ellipses place nodes; they are NOT exact physical modal branches for
general nonproportional tensors. Single-medium node placement remains byteexact.
Analytical static contrast restoration and independent radial/azimuthal gates
retain original thresholds/orders. Native paired/single polynomial, projector
and propagation primitives are byteexact;391 other primary modules are exact.
The independent reference helper's optional return_mp preserves its entire
default implementation through a source inverse; paired subtraction is MP100.

One original GPU1/installed-Core run, session74604, is drained CLOSED0.
All413 frozen before/after/current SHA match. Execution-known runner2102817 is
absent outside sandbox; unobserved short-lived pytest PID remains honestly null.

Four NEW targeted pytest controls PASS/XML24.112s. Two new
36MHz diagonal matched SPD samples epsilon=mu=(1.4,1.1,.85) outside and
(2.7,1.9,1.35) inside, queried at (.052,.018,.061)/(.049,-.021,.063), have
independent analytic MP100 regular-matrix differences formed before conversion.
All8 separate EE/EH/HE/HH .005 gates pass, maximum relative error
1.234425903045773e-09. Reference arrays are saved
before native evaluation and native arrays once afterward. Radial/azimuthal
adaptation converges at64/64 with rtol2e-4/atol1e-10, maxima256/128 unchanged.
These queries span TWO height bins (-5,-4). Final-grid reuse visits both cached
groups with every compiler forbidden and reproduces the saved output exactly;
no reference/adaptation replay. Actual unique retained GPU storages total
255960832B below the unchanged268435456B ceiling.
Gradient-flag mutation without version bump refuses both cache/direct action.
This is retained-cache accounting, NOT whole setup/action/solver/allocator peak.
Tail exponent batches64 nodes; construction/output/workspace remain unqualified.

Two NEW equal/unequal-ellipse radial controls compare three exponentially
weighted radial moments against independent analytic MP100 factorial values
at1e-9. A NEW strategy/configuration control checks exact public payload
roundtrip, early incompatible-map and periodic refusals, strict1e-20 loss
refusal, and exact-zero spatial contrast without any plane compiler. These
controls certify their selected numerical/contract scope only. The prelaunch
audit binds the preserved410-input preaudit copy; final413 launch additionally
binds that copy and the independent audit JSON/Markdown. No prior successful
plane/single-spatial/public matrix/solve/producer/archive/planner/resource/full
regression is replayed. No LIVE provider is changed or restarted.

Near-zero spatial material cancellation, grazing/low-wavevector sheet handling
and general nonproportional spatial convergence remain unqualified. The old
public tensor AC grazing failure had no saved failing grid; these new queries
do NOT prove its recovery. The scalar-limit cell source convergence failure,
original covariance failure, both f30 capacity failures, growing resource
failure and all other physics failures remain preserved. Original all-point
scaling and magnetic DAG jobs remain separate immutable evidence producers.
Public tensor AC/HB and exact persistence still require their own completed
producers and targeted cross-repository consumers. General complex, trainable,
nonlinear and periodic tensor domains remain required complete Goal work.

This completes the selected paired spatial implementation dependency only.
GSP01 stays OPEN and Goal ACTIVE; no capability receives production qualification.
Independent PHASE audit precedes human-owned phase commit. Root plan/checkpoint
append once after merge. Full current four-repository regression and complete
physics/complexity/resources/population acceptance, plus at least two independent
FINAL Goal audits, remain required at major/final merges with unchanged scope
and thresholds. No complete Goal acceptance or FINAL audit is claimed here.

### Complete original 24-point FMM resource series summarized

The ORIGINAL preregistered series has terminated: handle15385 drained with rc0;
runner340785/coordinator340962/last worker2072475 are absent outside sandbox.
All424 frozen before/after/current dependencies remain exact. Its numerical
provider is immutable1204329519c3eb83512613ccc0e88ba7650ea78d, NOT the current
optimized/binary-center provider. No new measurement, order search, original
reference, public producer, archive or covered PASS was replayed for this summary.

All24 declared measurements are present: fixed-ka and growing-ka, every
60/240/960/3840-edge size, three fresh-process repetitions each. All declared
native adjoint/current-gradient gates pass at1e-11, maximum errors respectively
3.1169401820772255e-15 and7.279309742822276e-17. All individual repetitions and
sizes enter each ordinary least-squares log fit; no tail-only or omitted-point
fit. Selected orders6/6/6/11 for fixed-ka and6/6/6/12 for growing-ka are reported;
closed original order receipts/repetition0 selections are reused only as
preregistered. Setup includes heterogeneous original admission reuse/search.

Retained-byte exponents are1.550622/1.579059. Forward-time exponents are
2.061161/2.156868, adjoint2.082100/2.074111, and backward1.632590/1.698119
(fixed/growing). These empirical results do not support a linear-complexity FMM
claim. Shared GPU timing interference is uncontrolled; measured finite-size
slopes are descriptive, not asymptotic proof or a new acceptance tolerance.

At3840edges, all three fixed-ka repetitions retain585948608B with whole interval
allocated/reserved peaks751039488/803209216B; growing-ka retains668235432B with
peaks853474304/868220928B. These six points pass comparison with the same-N
dense matrix byte count943718400B. Every repetition at60/240/960edges FAILS
that comparison:18 points retain their failures. Only a sampled memory bracket
between960 and3840edges is observed; this is not an exact crossover, measured
dense/H2 end-to-end crossover, or whole-solver resource qualification.
The interval starts before setup and includes all warmups/actions, differentiable
forward, first current backward and allocator-reset gaps. It excludes solver,
host memory, teardown and subsequent original reference checks.

Only cases0/12 have complete14400-coefficient original weak-equation references;
their global/cross-body errors1.9276165973931936e-16/2.1982062274835266e-12
pass1e-8. Both are the same physical size/frequency declared in the two families,
not two independent physical cases. This is consistency with implemented
quadrature, not independent electromagnetic or quadrature-convergence acceptance.
Other22 cases have no all-coefficient original reference; preserve null values.

Documentation only is merged against current3f16a58e3289506856f31896d02a7c759bb0d71a;
all396 current production modules remain byte-exact. Old measurements are never
relabeled as current optimized source qualification. GSP01/GSP09 remain OPEN,
all prior failures remain, and no capability obtains production qualification.
Independent PHASE audit precedes the separate human-owned stage commit. Root
plan/checkpoint append ONCE after merge. Goal ACTIVE; original final scope and
thresholds remain unchanged. Complete current four-repository regression,
physics/complexity/resource/population acceptance and at least two independent
FINAL Goal audits remain required at major/final merges.

### Exact original reciprocal near recomputation primitive completed

One appended StreamedReciprocalScalarNearChannels class in primary packed_near
reuses every old packed function/class byte-exact. Other395 primary modules
are unchanged against fdaec4352ced4e2b0c0329a5c75d13cef36ee7f3. The internal
CUDA complex128 scalar primitive retains validated reciprocal CSR/J-M maps
with zero-sized coefficient carriers. Each action recomputes original D/K
integrals in bounded chunks; both off-diagonal directions use the same fresh
values with ordinary transpose, and diagonals integrate all entries. No fitted
near substitution, persistent numerical cache, physical pair loss, tolerance
relaxation, material-loss removal or CPU production path is added.

Explicit borrowed physical tensors bind identity/version/gradient flags; an
owner guard runs before integration. Factory, medium, integration, maps and
coefficient budget are guarded. Fixed finite native D/K matrices cannot hide
larger backing storages. Three coefficient-array allocations are budgeted,
batch sizes shrink to fit, and an oversized single block refuses BEFORE
integration (possibly after allocating maps). This is coefficient-chunk
accounting, NOT original-quadrature/gather/product/allocator/solver/host peaks.

ORIGINAL4365 drainedrc0, runner2930131 absent outside sandbox; short-lived
pytest PID was not observed and remains honestly null. All414 frozen
before/after/current SHA matched at original closure;13 NEW pytest PASS,
XML5.645s. Post-closure edits to FEATURE/local plan only replace pending status
with this completed primitive evidence; numerical source/test are unchanged.
Their pre-edit bytes are preserved, not falsely called414currentstable later.

A NEW irregular tetrahedron with6 RWG edges,137MHz,
epsilon2.35/mu1.47/sigmaE.006/sigmaM.013 and q4/singular3/pair16/far32 uses
ONE complete12x12 original weak matrix generated once and saved before actions.
Nonascending1/2/3-edge maps preserve all9 logical and6 canonical pairs.
Every column and15 noncontiguous RHS pass original1e-8 consistency gates:
columns3.7107713379088457e-16, forward3.9016101252311267e-16,
original H3.8381975029846896e-16. Native H1.139349209158527e-16,
current gradient0 and higher-current gradient2.4599039823778547e-16 pass1e-11;
higher gradient uses saved native columns, not a stronger assumption about
the original1e-8 reference. Fixed-current forward saves zero internal tensors.
Constructor factory calls0, all required actions/backwards48 canonical calls.
Actual owned maps584B omit800B of retained original coefficients for THIS
fixture only. Reference/native arrays are saved for dependent controls and
must not be recomputed. Original weak consistency is NOT independent EM
or quadrature-convergence acceptance and proves no f30 fit or wall time.

A separate NEW same-shape synthetic control uses three two-edge groups,
actionbatch8/budget384B, and observes all9 emitted chunks reduced to1 entry.
Exactly6 unique canonical factories supply both directions; all synthetic
columns match an independently declared closed-form D/K matrix at
4.982999267486073e-21 <1e-11. This directly checks the bounded chunk policy,
not physics or allocator peak. Eleven new physical/factory/map/budget mutation
and malformed-output guards pass. No old successful reference, planner,
archive or packed suite is replayed. v1 preaudit404 and v2410 input copies
remain exact; only an additional NEW control was appended before the ONE
numerical run, preserving all original12 bodies and all production code.

This is an internal primitive. FMMConfig/capacity/regional/system/public solver
and exact persistence integration are the next dependency and remain OPEN.
The cost of repeated original quadrature has no wall-time qualification.
Old f24 physical/f30 capacity failures remain. Original binaryf30/public tensor
AC/automatic magnetic DAG jobs remain LIVE and immutable. No capacity,
whole-resource, complexity, general-material or production qualification.
Goal ACTIVE;GSP01/GSP09 OPEN. Independent PHASE audit precedes a separate
human-owned stage commit, then root plan/checkpoint append once. Complete
current four-repository regression, physics/complexity/resource/population
acceptance and at least two independent FINAL Goal audits remain required
at major/final merges with original scope and thresholds unchanged.

### Public exact original reciprocal near recomputation completed

Five existing primary modules integrate the previously committed streamed
scalar near primitive into explicit FMMConfig, structural capacity, regional
and system construction, and public Scene -> Simulation -> Result. Canonical
version cartesian_fmm_streamed_reciprocal_dk_config_v1 binds the positive
near_coefficient_workspace_budget_bytes. Six affected old schema comparisons
target only the added field's omission and parsing; old payloads/construction
remain exact. Periodic/complex64 and inactive budgets refuse. No constitutive
loss, physical pair, integration rule or acceptance threshold is changed.
The committed packed primitive and391 other primary modules remain exact.

Canonical reciprocal maps retain no near D/K coefficient tensors. Original
regional D/K integration is recomputed per action, with the same fresh block
for ordinary-transpose reverse coupling and complete diagonal integration.
Actual basis/geometry/tree owner guards and borrowed fixed CUDA inputs remain
bound. Structural original retained/table/near/projection gates remain; a
single coefficient block refuses before moments, possibly after automatic
order selection. The coefficient budget counts three bounded channel arrays,
NOT quadrature/EM/gathers/products/whole allocator/host/solver peaks or time.

ORIGINAL25677 drainedrc1, runner3326172/pytest3326401 observed LIVE then absent
outside sandbox; all423 before/after/current frozen SHA matched at closure.
Seven NEW controls yielded six PASS and one invalid guard-test fixture FAIL.
The whole original test/five primary modules/XML/raw references/results were
preserved BEFORE a failed-only successor. Only the failed metadata control's
public refusal calls receive required local Riesz storage; existing GMRES and
fallback remain unchanged. All other six complete bodies and all production code stay exact. Successor
39267 drainedrc0/one PASS,439 frozen SHA exact. Its short-lived
pytest PID was not observed and remains null; known runner3426891 is absent
outside sandbox. The six PASS controls, weak
reference, producer, planner and original primitive suite were not replayed.

The NEW48-support irregular tree uses ONE physical partition/planner call,
with independent integer canonical map/CSR inventory:3464B near maps,
1536B omitted theoretical original coefficient arrays and48 canonical pairs.
Four NEW action/near/projection/coefficient budget controls refuse before
numerical moments. These inventory/refusal checks are not physics acceptance.

ONE new lossy magnetic public fixture uses radii.008/.011 at
(-.253,.027,.013)/(.247,-.023,-.009),epsilon1.9+.012j/2.8+.017j,
mu1.18/1.29,117MHz/sub0/q4/singular3/leaf8/separation.25/p6..12,
compactshared/separable/native-midpoint/B32/coefficient8192B. Original
kernel1e-6/1e-5, GMRES/rwg-diagonal/rtol1e-12/atol0/max240/restart120
and true complete weak1e-10/atol0 gates remain. A complete120x120 original
weak matrix is generated once and saved before actions. All columns/15
noncontiguous RHS original1e-8 checks pass:{"all_columns": 2.957010932484084e-16, "forward": 6.697529806321197e-16, "adjoint": 4.879860104781137e-16, "native_H": 3.405558513173176e-16}.
Native H/current/higher-current1e-11 checks pass: current
4.2224482007955195e-17, higher 5.901888222905618e-16.
Higher gradients use saved native columns; forward saves zero internal
tensors. Actual owned bytes947952 equal the structural
plan, all three near owners retain zero kernels, minimum order6 passes the
original truncation budget. ONE actual public solve complete original weak
residual4.918081415909506e-13 passes1e-10/atol0. Exact Result
load restores currents/config under forbidden constructors/solves, and
coefficient-policy tampering refuses. The saved reference and producer
SHA3534868240791199541cf88453f002dffd908894b64f72bf5d52089b1b8956c8 must be reused, not regenerated.

ONE Bench consumer 71218 drainsrc0/one PASS,486 frozen SHA
exact; short-lived pytest PID was not observed and remains null, known runner
3430537 is absent outside sandbox. All69
Bench primary modules are unchanged and the protected untracked original
test stays SHA65e89f190547bde62175a865390e6a0b0c6910981e762cda935b5d7cb2953860.
Independent physical/numerical declarations precede opening ONLY this closed
producer. One lifted-state write/read checks native schema11/exact J/M bytes,
canonical config/ownership and altered budget/policy identities, with all
constructors/solves forbidden throughout. No producer, reference, planning,
field query or previously successful archive is rerun. This proves exact
storage/current consistency, not an independent electromagnetic oracle.

Seven distinct NEW Maxwell controls and one NEW Bench consumer pass across
the original and failed-only successor. The original failure remains recorded.
Postclosure edits correct only that test and completed FEATURE/docs; closure
identities retain their actual frozen bytes. The original public fixture elapsed1291.5108590498567s under shared
GPU use; this is not controlled timing or wall-time qualification.
Whole physical/quadrature/time/
memory/resource/asymptotic qualification and f24/f30 failures remain OPEN.
Other original binaryf30/public tensor AC/automatic magnetic DAG providers
and jobs remain immutable. No production qualification or FINAL Goal audit.
Independent combined PHASE audit precedes separate human-owned Maxwell/Bench
commits, then root plan/checkpoint append ONCE after merge. GSP01/GSP09 OPEN,
Goal ACTIVE; full current four-repository regression, complete physics,
complexity/resources/populations and at least two independent FINAL Goal
audits remain required at major/final merges with full scope/gates unchanged.

### Directional exact original reciprocal near recomputation completed

Five existing primary modules integrate scalar_reciprocal_streamed_dk_v1
into explicit DirectionalFMMConfig, capacity, automatic sparse/DAG region
construction, original regional near actions and system construction.
Canonical automatic_directional_streamed_reciprocal_dag_fmm_config_v1 binds
the positive near_coefficient_workspace_budget_bytes; old directional payloads
omit the new field exactly. Cartesian prefixes, the committed packed primitive
and391 other Maxwell primary modules remain exact. Physical partitions, FAR
builders, carrier/order rules, constitutive losses and acceptance gates remain.

Canonical physical reciprocal maps retain zero D/K coefficient bytes. The
same original integrals are recomputed in bounded action chunks, preserving
ordinary-transpose reverse coupling and full diagonal integration. Actual
basis/geometry/tree guards remain bound to borrowed CUDA inputs. Single-block
coefficient and retained-map limits refuse before sparse maps or DAG FAR
construction. Action workspace includes the three bounded coefficient arrays;
quadrature, EM products, allocator/host/solver peaks and runtime are unqualified.

Original Maxwell handle40017 drainedrc0; runner3489438/pytest3489675 were
observed LIVE then absent outside sandbox. All424 frozen dependencies matched
before/after/current at closure. Seven NEW controls PASS/XML43.305s, elapsed
45.22275814320892s with shared GPU use, not controlled timing.
NEW48-support full integer leaf-pair inventory has2304 logical/1176 canonical
pairs,66632 map bytes and37632 omitted theoretical coefficient bytes. One
capacity call covers that domain. Four NEW sparse/DAG map/coefficient budget
refusals occur before allocation. Existing successful planners are not replayed.

ONE NEW public three irregular tetrahedra fixture,173MHz/36 currents, carries
independently declared lossy electric and magnetic media, q4/sg3, leaf3/.15,
quadrature12/p0..12, RHS4,4096B coefficient budget and local Riesz GMRES.
ONE36x36 original weak reference is saved once; all columns and15 noncontiguous
RHS yield columns1.9930220257577926e-15/F1.9457480629429557e-15/
originalH2.1902224059067754e-15 below1e-8. NativeH1.9559057406664961e-16,
current0/higher3.9568288813765173e-16 below1e-11; saved forward tensors0.
Maps704/240/240/240B retain zero kernels. Borrowed native inventory571096B
is not whole peak or complexity qualification. Exterior FAR6, interior FAR0
for all three bodies, positive changing links0: no high-frequency/changing
carrier/interior-FAR qualification. ONE public solve complete original weak
residual1.322525588871928e-14 passes unchanged1e-10 relative/0 absolute gate.
This saved-fixture check is not an independent electromagnetic oracle or a
general certificate for sampled runtime residuals. ONE result SHA
162644b66f87beb611925c4d0c8cf2a0a126c5af9f242e074c392a4683c84136
and saved native reference remain reusable without reconstruction.

Bench prelaunch v1 audit found missing actual J/M byte comparisons before
consumer execution. Whole old test,484-input snapshot and audit were retained.
Only two byte comparisons were inserted, with exact whole-test inverse proof;
incremental v2 audit cleared P1/P2 and490 identities. No numeric failed run.
One consumer original handle78454 drainedrc0/XML5.978s, all494 before/after/
current SHA exact. Runner3525489/pytest3525908 were both observed LIVE then
absent outside sandbox. Independent complete Scene/config declarations precede
opening only the closed producer. Native schema11/exact J/M, one lifted-state
write/read and eight changed config identities pass under constructor/solve
bans. All69 Bench primary modules and protected tensor test remain exact.
No producer, reference, solve, field query or covered successful check replay.

Completed documentation is edited only after both actual closures; frozen
identities preserve actual closed bytes. Independent combined PHASE audit
precedes separate human-owned Maxwell/Bench commits, then root plan/checkpoint
append ONCE after merges. GSP01/GSP09 remain OPEN and Goal ACTIVE. Complete
physics, whole resources/runtime/complexity, populations, current four-repository
acceptance and at least two independent FINAL Goal audits remain required at
major/final merges. Original scope and gates remain unchanged; no production
qualification, final Goal audit or full regression is claimed by this phase.

### Exact binary lattice centers merged with current configuration contracts

Four existing primary modules add explicit binary_lattice_v1 actual expansion
centers on a common power-of-two grid with integer magnitude bound2**51.
All seven original tree topology/support tensors remain exact; coefficients
use actual native center differences, without key projection or pair deletion.
Complete original support corners set radii with one outward ULP; this is not
a formal interval certificate. Same-center distinct nodes retain physical near
interactions and cannot enter a nonzero displacement catalog.
Canonical cartesian_fmm_binary_lattice_config_v1 binds the center policy;
free-space/float64/complex128 guards and constitutive losses remain unchanged.

Original geometry/public handle59569 drainedrc0, six distinct NEW controls
PASS/XML94.802s with422 stable inputs; runner2149757/pytest2150090 were both
observed LIVE then absent outside sandbox. Original Bench81698 drainedrc0,
one consumer PASS/XML3.742s with488 stable inputs, runner2432798/pytest2433103
both observed LIVE then absent. Prelaunch contract corrections and whole
original proofs are preserved; no numeric v1 rerun. The NEW120-support geometry
has121 nodes/3596 FAR blocks,2610 exact integer/native displacement owners,
all physical pairs exactly once and four explicit same-center interactions.
ONE105MHz two-sphere public fixture builds ONE120x120 original weak reference
and one result SHA904beb5e4f68e46ce612c5c3693ea50bac201e0c6ce08e15f23cc41233762a02.
Selected order7 follows the original kernel gates. Original column/F/H,
native adjoint/current/higher gradients pass unchanged1e-8/1e-11 gates; no
forward internal tensors saved. Complete weak residual1.8805009051738983e-11
passes1e-10/0 absolute. Owned1454112B matches original capacity plan.
This is saved original-weak consistency, not independent electromagnetic or
whole resource qualification. One original Bench consumes this result once,
exact schema11/J/M, independent Scene/config and tamper guards under banned
constructors/solves. Neither this reference/solve nor this archive is replayed.

Original f30 handle87713 drainedrc0, runner2204217/pytest2204690 observed LIVE
then absent outside sandbox;422 before/after/current identities exact at
closure. XML6175.192s/one instrumentation PASS records an actual capacity FAIL,
not admission. The same original failed resonant sphere R.1/f414635862.3319615/
epsilon12+.01j/mu1 has27000 edges/54000 currents/18000 faces, leaf8/.25,
p6..12, original positive_tail_min_v1 kernel1e-6/1e-5, reciprocal stored scalar
near, compact shared table/separable shifts and original2GiB total/1GiB
component budgets. Two original12197-node/9656-leaf trees yield p6 total
10239088304B FAIL, table6421642080B FAIL, near3520407824B FAIL, projection
292271360B PASS. No order search, expansion, quadrature, action, solve or fields
followed this refusal. Elapsed6171.52272933349s under shared GPU is not timing
qualification. Whole14 actual input/test/primary/XML/raw/receipt files saved
before any union development, original f24/dyadic/midpoint failures retained.

After all three original jobs terminated, a new isolated current union applies
only the closed binary four-module replacements to current Maxwell baseline
8b109802704cbe058ab9974e67d0693142166896. Every whole old binary module is
reconstructed exactly, and every union module inverses to its whole current
baseline exactly. Other392 current Maxwell primary/all69 Bench primary remain
exact. Old tests/plans are copied byteexact; old providers remain immutable.
ONE NEW config-only current merge control original handle68567
drainedrc0/XML4.365s with575 frozen inputs exact
before/after/current at closure. Observed runner/child state is recorded in
the actual terminal receipt. Original79359 drainedrc1/XML3.270s,548 exact
closure: the fixture passed a serialized kernel-budget dict directly to its
typed constructor. Whole actual failed test/four primary/input/XML were saved
before a failed-only copied control changes only that line to explicit canonical
configuration restore. Production and all declarations stay exact. Short original
runner/pytest were not observed LIVE (pytest null), known runner3558594 absent;
no PID was invented. The original failure remains archived. Only after successor
closure is the corrected whole test copied into the union. Pinned old binary payload restores exactly while
omitting the newly added coefficient budget. New binary+streamed payload binds
all current controls, changed old schemas refuse it, and directional streamed
dispatch remains exact. This control performs no geometry/planner/reference/
producer/solve/archive replay. Combined binary+streamed ACTIONS are not qualified.

Completed FEATURE/plans/assessments change after actual closure; their closed
bytes are saved. Independent PHASE audit precedes separate human-owned
Maxwell/Bench commits; root/checkpoint append once AFTER both merges. Six
closed NEW Maxwell controls plus one NEW current config control and one closed
Bench consumer pass, without successful covered check replay. One original
f30 instrumentation PASS preserves capacity FAIL. Scope and thresholds remain
unchanged. GSP01/GSP09 OPEN/Goal ACTIVE; independent physics, complete
resources/time/complexity/populations, current four-repository acceptance and
at least two independent FINAL Goal audits remain at major/final merges.
No full regression, production qualification or FINAL Goal audit is claimed.

### Exact original scalar M2L signed-axis sharing primitive completed

Two existing primary modules append an internal exact-native signed-axis
catalog and compact translation view/batch primitive. Their whole old prefixes,
including opposite/equal keys, compact coefficients, table builder, native
batch and fixed-current autograd remain byte exact;394 other current Maxwell
primary/all69 Bench primary unchanged. No public configuration, physical
partition/order/integration/medium/loss/acceptance gate changes.

Keys contain exact IEEE sorted absolute components of actual native physical
displacements. Owners are original physical displacements, never projected or
rounded. All original input pairs retain separate signed/permuted bindings;
zero displacements refuse. A view borrows the original owner/layouts without
new retained CUDA tensors. Taylor derivative multi-indices and reflection
parity transform Green coefficients; gradient components include the extra
axis sign. Physical electromagnetic moment channels stay in original global
coordinates. Owner/medium/version/gradient flags/axis controls are guarded.
A batch generates bounded temporary actual-axis jets and reuses the existing
native action. A conservative visible≤128MiB gate includes extra temporaries;
allocator/library/host/solver/quadrature/whole resource peaks remain unqualified.

Prelaunch v1 P2 found finite-mask allocation before workspace/current refusal.
WHOLE original appended source/test,479-input snapshot and audit were retained.
Only the new finite check moved after admission, with mask/jet tripwires plus
one saved-forward-owner-version/backward refusal control. Exact whole inverses
and original prefixes remain. No numerical tests had run before this correction.
Prelaunch v2 mistakenly normalized ABS/REL test paths before validating all
entries, hiding one old absolute test SHA. Root preparation correctly rejected
the lexical mismatch, but root orchestration failed to check its exit code and
still called the launcher. Actual ORIGINAL44361 drainedrc1, runner3700042 absent
outside sandbox, no pytest/receipt/XML/run.log/reference/producer existed.
This is an actual pre-pytest identity failure, not a numerical test FAIL or PASS.
The whole13 current source/test/input/execution/audit files and actual terminal
were preserved before any identity correction. Historic v1/v2 reports are
unchanged; the auditor explicitly acknowledged alias-normalization omission.

NEW v3 collector updates only that frozen absolute SHA; production and tests
remain exact. All511 lexical entries were checked before alias normalization;
v3 preaudit P1/P2 empty. Checked exit-zero preparation then froze515 entries.
Original numerical handle74901 drainedrc0, runner3733692/pytest3734169 both
observed LIVE then absent outside sandbox. All515 lexical before/after/current
hashes exact at closure. Eleven NEW controls PASS/XML27.763s, elapsed
29.38601983524859s under shared GPU use; no controlled timing qualification.

NEW1057 logical catalog inputs retain49 distinct physical input pairs (repeated
inputs are declared, not a physical partition coverage claim), two1024-sized
chunks/two original owners, all48 signed permutations and one-ULP nonsharing.
Orders0/4/7 with row batches1/7/11 independently construct48 original native
compact references once per order, save each reference before view actions,
and check every Green/gradient table entry,15 noncontiguous RHS, original F/H,
native H, current/higher derivatives and zero saved forward tensors at1e-11.
All48 views use two32/16 translation batches with RHS3. Overall maximum error
5.723456203101935e-14 passes; this does not validate highest order/RHS resources.
Owner104/4760/16960B is shared by48 views with zero additional retained bytes;
48 independent owners would retain4992/228480/814080B respectively. This
comparison is not a claim of a48-fold gain versus the previous opposite/equal
catalog, nor actual f30 capacity qualification. Six mutation controls cover
axes/signs/alias/version/gradient flag and saved-forward/backward mutation.
Strict binding, zero/invalid physical pair and preallocation refusals pass.

Three saved native reference SHA are
885bd9db4bbd6732875564acf865def6cf74c7c0bdd4d2fef88da4c0de4c8994,
909551d0be962be79d9416fe05e68aa0a1def0ffeb5219a0e956cbd6585507ac,
3ae00f8a581bd0d5062771192e059c64654c5a679870b00f23832807068849e3.
They must be reused for any failed-only successor, not regenerated. Original
kernel algebra is not an independent electromagnetic oracle. Public integration,
f30 actual capacity/physics, periodic/tensor/general materials, all-N resources/
complexity and production qualification remain OPEN. Original f30 table
6421642080B/near3520407824B/total10239088304B failures remain unchanged;
no original successful planner/reference/public producer/solve/archive replay.

Completed FEATURE/plans/assessments change only after actual closure, retaining
closed bytes. Independent PHASE audit precedes a human-owned Maxwell commit
and merge; root/checkpoint append ONCE afterward. No Bench commit is needed
for this internal primitive;69 Bench primary/protected original tensor test
stay exact. GSP01/GSP09 OPEN, Goal ACTIVE, original scope/gates unchanged.
Full current four-repository acceptance, complete physics/resources/time/
complexity/populations and at least two independent FINAL Goal audits remain
at major/final merges. No full regression or production qualification claimed.

Before executing the original planned merge, root identified that the borrowed
view had inherited owner-only factory/reverse methods; the inherited reverse
constructor would build a view missing its required owner binding. This is a
source-proven protocol defect, not an invented numeric FAIL. The entire closed
eight-file phase, original final audit and manifest were saved before correction.
The original phase audit had missed that inheritance exposure and is retained.
Only the new view protocol changes to explicit composition: apply/adjoint/
tensors/action alias the identical original functions; state/current checks
call the identical original base helpers. Owner construction methods are no
longer implicitly part of a borrowed view. No table/jet/batch algorithm or
acceptance gate changes; exact whole inverse and original prefixes remain.

One NEW protocol control declares the same original211MHz/material/order4/
displacement plus new RHSseed22417 before opening the pinned saved p4 reference
SHA909551d0be962be79d9416fe05e68aa0a1def0ffeb5219a0e956cbd6585507ac.
Taylor coefficient generation is forbidden; one owner takes saved coefficients
and its original small layout. Native forward/H/current/higher derivatives,
bounded F/H batch, zero saved forward tensors/extra owned view bytes and
saved-forward-owner-mutation refusal pass unchanged1e-11 gates. Maximum
1.7055387774783185e-15. Original handle19663 drainedrc0,
one PASS/XML4.883s; all502 lexical hashes exact before/after/current
at actual closure. The short-lived runner3779208 was not observed LIVE and is
absent outside sandbox; pytest PID remains honestly null. No child is inferred.
The original eleven PASS tests and three original reference constructions are
not replayed. Twelve distinct NEW controls therefore pass across two source
epochs; original515 closure and502 protocol closure retain their actual bytes.

Final scope is nine Maxwell files, no Bench changes. Completed feature/docs
are updated after this second actual closure. Revised independent PHASE audit
reviews only the source protocol delta, completed combined evidence and final
merge plan, retaining the original audit. Separate human-owned commit then
root/checkpoint append once AFTER merge. Whole resource/highest endpoints,
public integration/f30/independent physics/production remain unqualified;
GSP01/GSP09 OPEN and full unchanged Goal ACTIVE.

### Scalar GMRES finite precision residual restart completed

One existing Maxwell primary scalar GMRES routine now permits a residual restart after finite precision closes the Arnoldi basis only when the cycle reduced the actual unweighted true residual, the original stagnation window has not fired, and original maxiter remains. No tolerance, iteration budget, restart size, residual weighting, twice reorthogonalization, CUDA least-squares/direct-fallback policy, zero-progress/nonfinite/failed-factor guard or block GMRES changes. All395 other Maxwell and69 Bench primary stay byte exact.

The dependency arose from actual public signed-axis original72705 rc1/1FAIL/XML475.137,638 lexical source hashes exact:72iterations/145matvecs, inner breakdown1.2445038476312128e-12>1e-12; complete original weak4.99689875981052e-12<=1e-10 cannot substitute for inner convergence. Actual whole state SHA b9d6c45f1df8bc5b61c78d0d5fa4c3f5cd88332077cde417b421917fe0445367 retained. No completed public result or Bench consume; all historical unpreconditioned/leaf failures preserved. Other original LIVE providers are untouched.

New original92601 drainedrc0/4PASS/XML4.182s,410 frozen lexical hashes exact at closure, runner4176852/pytest4177145 observed LIVE then absent outside sandbox. Three NEW controls plus one affected existing inconsistent-system guard. Same structured native CUDA triangular matrix and exact representable complex current/RHS, frozen whole original scalar function comparison, no direct solve: complex64 old breakdown relative1.4722621784872065e-7 versus rtol1e-8, new convergence0 after7iterations/one restart; complex128 old breakdown2.779113003847485e-16 versus rtol1e-16, new9.016633219342427e-17 after5iterations/one restart. Exact native matrices/old-new solutions/RHS/history/state SHA saved before assertions. NEW maxiter4 budget remains breakdown with0restarts; affected inconsistent system remains failed. This is a scalar numerical control, not an independent electromagnetic oracle or public solve qualification.

Whole collector sources and original primary/inverse proof retained. Separate human-owned commit follows independent PHASE audit; root/checkpoint append once after merge. No previous successful physics, planner, reference, public action or archive replay and no per-edit full regression. Public signed-axis failed solve must continue against this dependency after merge. Original f30/full physics/resources/runtime/complexity/current four-repository acceptance and at least two independent FINAL Goal audits remain OPEN. Goal ACTIVE/GSP01 OPEN; original objective and gates unchanged, no production qualification.

### Public exact signed-axis scalar M2L sharing completed

Nine existing Maxwell primary modules integrate exact original signed-axis coefficient sharing with a full explicit public configuration schema, binary-lattice centers, bounded original-axis CUDA batches and streamed original reciprocal near integration. The compact primitive/other387 Maxwell/all69 Bench primary stay byte exact. Every original physical support pair remains; support/kernel/order/integration/material/stopping/qualification gates are unchanged. Native float64/complex128, free space, order0..12, rows1..16, B1..32 and RHS4 are explicit. Visible128MiB includes temporary actual-axis jets, including B1 routing; whole host/device/solver/allocator/quadrature/workspace peaks remain unqualified.

Actual v1 original9160 rc1:6PASS/2FAIL/XML7.699 with485 each frozen hashes exact. One negative configuration fixture omitted explicit iterative stopping. Four original coarse tetrahedra exposed a zero partition-center span in valid full RWG supports, refused before reference or coefficients. Whole396 primary/actual test/input/audit saved before correction. The existing binary-center branch now preserves its unique original leaf/bounds and selects an actual binary center from complete original support bounds; positive-span call/topology and original support-corner radius/native integer gates unchanged. NEW geometry control verifies all bounds/corners/native grid and truly collapsed support refusal. Preaudit v2 found only the new oracle Pythonint ldexp exponent; whole five files/input/audit saved before Tensor exponent correction, no v2 numerics.

Actual v3 original18470 rc1:2PASS/1FAIL/XML6.924 with522 lexical hashes exact. Configuration/zero-span PASS are not replayed. Public preparation succeeded but FAR was empty; actual coverage assertion failed before any reference/solve. Whole actual inputs retained; whole396 actual primary reuse existing byte-exact snapshots without repeated archival. Only the still-failed public fixture coherently splits one original edge and its two adjacent faces. Same closed physical four-tetra boundary/positions/material/magnetic loss/143MHz/kernel/separation.2/integration/stopping/gates; actual discretization changes48 to72 currents. No same-matrix claim. Independent Bench declares this refinement before artifact open.

Actual v4 original17374 drainedrc1/1FAIL/XML148.033 with545 lexical sources exact. ONE72x72 original weak reference constructed and saved once. All columns/F/H/native Hermitian/current/higher gradients passed before an internal-regional B1 test incorrectly supplied requires-grad outside the public fixed-current wrapper. Whole partial/raw/reference retained before correction; no successful partial/reference replay.

Actual v7 original99810 drainedrc1/1FAIL/XML190.141 with583 sources exact. B1 versus B16 error1.9161503108888148e-17 passed1e-11; exact action-owned storage matched capacity before unpreconditioned solve failed. That epoch omitted actual failed solver/current/RHS state: observational gap remains explicit. Instrumented failed-only v9 original61182 drainedrc1/1FAIL/XML185.357 with612 sources exact preserved stats/current/RHS before assertion. Stagnation30iterations/61matvecs/native relative0.006661434371985221/complete original0.006661434371991041 fail unchanged thresholds. Whole state SHA7e43f11987a0ce39bd2e7428b2723b88c5a66d7521a7d5c5e59ba7c74914f537.

Existing public leaf_factor_kernel none->original was the ONLY v10 configuration change. Native original principal-leaf right inverse:20blocks/6240retained bytes, separate from FMM action storage. Actual v10 original72705 drainedrc1/1FAIL/XML475.137 with638 sources exact:72iterations/145matvecs/breakdown, inner1.2445038476312128e-12>1e-12; complete original4.99689875981052e-12<=1e-10 cannot replace inner convergence. Whole state SHA b9d6c45f1df8bc5b61c78d0d5fa4c3f5cd88332077cde417b421917fe0445367 retained.

Independent prerequisite c62477a3 corrects only scalar GMRES finite-precision Arnoldi closure: actual cycle progress, no original stagnation and remaining maxiter permit a true-residual restart. Three new CUDA controls plus affected inconsistent guard PASS, separately audited/committed. No tolerance/budget/restart-size/stagnation/fallback changes. Public closed worktree integrated its exact tree; nine pending public primary bytes unchanged. V11 configuration is EXACTLY identical to v10 including original leaf inverse. Same72-current four-body Scene/material/integration/143MHz/FMM action/order/capacity; rtol1e-12/atol0/maxiter240/restart120/stagnation20/1e-4/no fallback/original1e-10 absolute0.

Actual v11 original68239 drainedrc0/1PASS/XML507.816s with661 lexical source hashes exact at closure. Runner/pytest observed LIVE thenabsent outside sandbox. Actual12 physical FAR pairs/3 original coefficient owners/zero additional view-owned bytes. Prior successful cols/F/H/current/higher/B1/inventory reused without replay. Inherited errors{"all_columns": 4.1259403542484797e-13, "forward": 4.2936887037407434e-13, "adjoint": 4.2023066435713397e-13, "native_H": 4.743722529896882e-16}, current/higher1.4426073245419965e-17/5.0544911473909e-16, zero saved forward tensors, B1 error1.9161503108888148e-17. Action structuralowned1326624B equals prior plan and excludes separately receipted leaf factors/whole peaks. Actual solver iterations74/restarts1/inner relative6.559765508923764e-13; original admission checked. Complete original weak residual4.696234924074221e-12 passes1e-10. Exact Result restore forbids construction/solve. Producer SHA22e0d0c200b6d3b09dc56714e3ce7423f1db6345394eb2c1cb054fe4833fe751; original saved reference SHA837bcf6cf2f3870c11fee5926d840f0744aec3403a4c06ce17ded28a21aa2e2e. Original scalar algebra is not an independent electromagnetic oracle. Every earlier failure is preserved.

ONE Bench consumer original59769 drainedrc0/1PASS/XML4.858s with503 sources exact. Independent Scene/config before producer open, schema11/exact J/M, one lifted write/read, eleven metadata tamper controls and constructor/solve bans. All actual arrays including.npy are bound to evidence SHA. Protected original tensor test SHA65e89f190547bde62175a865390e6a0b0c6910981e762cda935b5d7cb2953860 unchanged. No reference or solve repeated.

Nine distinct new Maxwell controls supported across preserved epochs plus one Bench consumer PASS. Final single regression file assembles eight byte-exact actual-PASS controls plus the original full public function with only two internal B1 currents detached and existing original leaf inverse enabled. Final combined full function was not executed as a whole: algebra from v4, B1/inventory from v7, failed-only solve/restore from v11, with exact same FMM action construction; scalar GMRES dependency was independently accepted. Whole source copies and proof retained once before assembly, historical failed-only tests remain external evidence. Actual XML/receipt and every lexical historical byte are directly validated; complete396 oldprimary saved files and actual Bench state arrays included. Three recorder preaudit omissions and later immutable closure-pointer correction preserved in prior scripts/reports; conditional audits are not actual success reports.

Independent PHASE audit precedes separate human-owned Maxwell/Bench commits; root/checkpoint append once after both merges. No f30/independent physics/allN/complexity/whole host/device/solver/allocator peaks or production qualification. Original f30 table6421642080/near3520407824/total10239088304 capacityFAIL remains. GSP01/GSP09 OPEN, full Goal ACTIVE; original complete four-repository/final acceptance and at least two independent FINAL Goal audits still required. Scope and thresholds unchanged.

Actual evidence preparation v6 original65429 drainedrc1 (not a numerical collector): baseline Bench plan path was missing after the Maxwell union/docs and Bench feature/local entry were written. Whole partial files/script/audit retained once before correction; conditional preaudit path-existence omission preserved. Actual Bench plan is docs/plans/workspace/maxwell_bench_development_plan.md. Resume v7 only completes the missing Bench plan/assessment and final manifest; no union, archive, numeric reference, producer or consumer replay. Both baseline plan files are explicitly checked before final records.

### Bounded original reciprocal near coefficient cache primitive completed

The primary packed-near module now offers a bounded first-use native D/K cache composed with the original streamed reciprocal scalar implementation. The entire original module prefix and all395 other Maxwell primary files remain byte exact. Complete independent diagonal prefixes take priority, then canonical off-diagonal prefixes. Whole validated native blocks are copied; unselected blocks retain original integration. Plain reciprocal transposes, complete original diagonal integration, electromagnetic CSR action and fixed-current first/higher derivatives are reused unchanged. No approximation, fallback or altered acceptance gate is introduced.

The default cache budget is128MiB per regional owner. Planned native payload and exact maps are included in retained admission before cache allocation/integration, after existing map setup. Pending cache segments cannot serve reads. Every partial fill records an event; retries on another CUDA stream wait before overwriting an incomplete prefix, and ready reads wait for completion. Native per-block clones preserve original bounded factory backing storage. Physical/factory/map/cache/control mutation guards cover cold and warm actions. This is not yet wired into public FMM configuration, cumulative capacity planning or persistence. Original quadrature scratch, host/event storage, full process/device/solver peaks and performance remain unqualified; original action finite checks may synchronize implicitly.

Original targeted handle91510 was drainedrc0:36 distinct NEW cache-policy controls PASS, XML6.272s, all415 frozen lexical sources exact at closure, installed Core/GPU1 outside sandbox. Runner805193 is absent outside sandbox. Launch returned a running original handle; no separate runner LIVE snapshot or pytest child PID was collected, and none is fabricated. One SHA-bound saved137MHz irregular magnetic/lossy tetrahedron12-by-12 original weak reference was reused without regeneration. Native columns/forward/original-H errors3.7107713379088457e-16/4.764714780112304e-16/4.2093572525634516e-16 meet unchanged1e-8; native Hermitian/current/higher errors2.406945729403873e-16/0.0/2.7214713903187727e-16 meet1e-11. First fill makes6 original integrations; all later checked warm forward/H/current/higher actions make0. Saved forward tensors0. Exact retained maps584B plus coefficients800B equals1384B. Actual native columns/current/dual/output/H/gradient/higher state was saved before assertions.

Five new synthetic native budget boundaries,20 cold/warm mutation guards, three invalid cache budgets, retained-budget preallocation refusal, three invalid original field refusals, partial-fill cross-stream retry, completed cross-stream read and host prefix-plan control pass. Failed partial fill retains no readable prefix, retries6 original blocks and then makes0 warm integrations. No original13PASS controls, successful planner/reference/public producer/archive consumer, old failure cohort or full regression was replayed. Three unrelated original LIVE collectors remain untouched.

Independent PHASE audit and a separate human-owned commit precede dependent public integration. Root plan and checkpoint append once after merge. Goal ACTIVE/GSP01/GSP09 OPEN; all original failures, objective, final thresholds, four-repository full acceptance and at least two independent FINAL Goal audits remain intact and OPEN. No public/physics/whole-resource/runtime/complexity/production qualification or FINAL Goal audit follows.

### Public Cartesian bounded original reciprocal near caching completed

The accepted internal native-cache primitive is connected to the existing Cartesian FMM configuration, structural capacity, regional/system owners and public runtime. Five primary modules change; the primitive and391 other Maxwell primary files and all69 Bench primary files remain byte exact. Explicit scalar_reciprocal_cached_dk_v1 uses the strict cartesian_fmm_cached_reciprocal_dk_config_v1. Existing affected configuration payload/construction keys stay exact. Cache budget is128MiB per regional owner by default, including zero as an explicit option. Aggregate planned native payload enters original total and cumulative near admission before numerical moment/table/cache/integration setup. Shared host topology matches the original canonical grouped owner, independent diagonal priority and exact reciprocal map storage. No approximate channels, numerical fallback, physical geometry/material modification or tolerance relaxation is introduced.

Original public collector61879 drainedrc0/5 distinct NEW controls PASS, XML117.907s and all429 lexical hashes exact at closure. Runner896724/pytest896886 were observed LIVE and then absent outside sandbox. Two actual prelaunch test-contract P2 findings were retained with every whole lexical input before correction: late reference presence/identity checking and nonfinite failure-JSON retention. Only those test contracts changed; production bytes stayed fixed. The successor requires reference presence/SHA before setup, independently declares Scene/config before opening it, preserves complete native .pt states before assertions, and explicitly requires finite errors/residual with original gates.

The new cached public fixture preserves the original four-body lossy magnetic split-tetrahedron Scene and declares fixed p7, the original admitted actual order. The covered successful p6..12 order search is not replayed; actual p7 M2L kernel gates1e-6/1e-5 still run. This targeted result does not qualify automatic selection. The original72-by-72 weak reference SHA837bcf6cf2f3870c11fee5926d840f0744aec3403a4c06ce17ded28a21aa2e2e is reused without regeneration. Native columns/F/H errors4.1259403542484797e-13/4.1180964633901093e-13/4.1418418218880557e-13 pass1e-8; native Hermitian/current/higher errors3.738090366796443e-16/2.0762538528414475e-17/5.423964540366681e-16 pass1e-11. Constructor near integrations0, first cache fill149 original integrations, all checked warm actions and solver make0 further near integrations, forward saved tensors0. Exact action inventory1341376B includes14752B native coefficients across five separate regional owners. These retained bytes do not bound all scratch, allocator, host/event or whole-process resources.

The unchanged local RWG Riesz, original leaf LU, GMRES rtol1e-12/atol0/max240/restart120/no fallback converge after74iterations/149matvecs/1restart, inner relative7.353280334121991e-13. Complete original weak residual4.70385042659551e-12 is finite and passes1e-10; the original checked sampled/dual error-budget gates also pass. Full actual native and solver states are preserved before assertions. One result SHA99bc92210b43499c8386d4edbf4678f311a3dd6b3513098f0b8a04ce65c3c7d2 restores exact currents with construction/solve refused and cache-budget tamper rejected. Collector timing is recorded, not a comparable runtime or peak-memory qualification.

One Bench consumer original18626 drainedrc0/1PASS, XML4.027s and all485 lexical hashes exact at closure. Its original driver records runner910643/pytest910811; both are absent outside sandbox. The first outside snapshot occurred after this fast consumer exited; no LIVE snapshot is fabricated. Independent physical/numerical declarations precede producer open; one write/read retains exact J/M and strict native schema11/cache budget identity, construction/solve bans and13 changed-configuration refusals. All10 actual .npy arrays and manifest are bound as evidence. No second producer, reference, setup or solve is run.

Independent PHASE audit and separate human-owned Maxwell/Bench commits precede merge recording. Root/checkpoint append once after both merges. No accepted primitive36PASS, old public9PASS or exact archive consumer, old planner/reference or full regression is replayed, and no prior failure is removed. Three unrelated original LIVE collectors remain untouched. GSP-01/GSP09 and Goal remain OPEN/ACTIVE. Full original physics/f30/all-N/crossover/complexity/automatic-selection/directional/general material scope, whole quadrature/host/device/solver peaks, current four-repository full acceptance and at least two independent FINAL Goal audits remain OPEN with unchanged thresholds. No production qualification or FINAL Goal audit follows.

### Directional bounded original reciprocal near caching completed

Five original primary modules integrate the accepted cache primitive into
Directional sparse/DAG and public Scene -> Simulation -> Result construction.
The new strict configuration version is
automatic_directional_cached_reciprocal_dag_fmm_config_v1. The nonnegative
128MiB default cache budget is PER REGIONAL OWNER, not a global or peak budget.
Exact original canonical group ordering determines selected complete native
D/K payload, admitted with maps before sparse/DAG projections and numerical
tables. Zero cache budget remains valid; original integrators, automatic
partition/cone/order selection, physical materials, original Riesz and
GMRES thresholds are unchanged. Other391 primary modules, the accepted cache
primitive and Cartesian construction bodies remain byte exact.

Original producer 95910 drained actualrc0/7 NEW PASS,
XML33.090s; all420 lexical source hashes exact.
Runner1631190/pytest1631373 observed LIVE outside sandbox, then both absent.
The same original173MHz three lossy magnetic tetrahedra and36 physical currents
reuse saved original36x36 weak reference SHA
bc999b41dbca5779ad484e00b457cbe3c9eb42562218502a16649305d8afe646.
Automatic plans rebuild once as a necessary new cached-policy construction
dependency; no old planner test or reference production is replayed. Complete
native column/F/H relative errors remain below1e-8; Hermitian/current/higher
gradients remain below1e-11, zero saved forward tensors. Original near cold6
factory calls, warm0 across native actions and solve. Native cache payload6912B
(exterior3456 plus three1152B interiors); complete retained native dependency
inventory578008B. That inventory excludes host topology, setup/action/backward
scratch, Riesz/solver and allocator/library state; no whole peak qualification.
Actual public GMRES convergence and original error admission pass; complete
original weak residual 1.322525588871928e-14 <=1e-10. All actual
native/solver states are retained before failure assertions. One sparse
zero-cache action reuses the SAME new root plan, does not reselect/reference/
solve, performs24 original integrations and matches DAG F/H exactly. Exterior
FAR6, interiorFAR0, positive changing-carrier links0: no high-frequency or
changing-carrier qualification follows. One new exact public restore succeeds
without construction/solving; changed cache identity refuses. Producer SHA
ab2b142f6402a308e0ffe45d4208424b2209a9c5e8fbf87080bdd8cb253134e8.

Original Bench consumer 56657 drained actualrc0/1 NEW PASS,
XML4.451s; all484 lexical hashes exact. Physical
and numerical identities independently declared before producer hash/open.
One write/read retains exact native J/M, native schema11 and Directional cache
budget identity; ten changed-configuration refusals and construction/solve
bans pass. All eight actual archive arrays are bound. No second producer or
reference/setup/solve is run. Outside LIVE PID observation status is retained
exactly in terminal-v1, including any fast-process observation gap.
The recorder's actualrc1 assumed ten arrays from the earlier four-body archive;
its whole script and failed-state manifest are preserved before correcting to
the eight actual three-body arrays verified against the archive manifest.
Both successful numerical epochs remain unchanged and are not replayed.

Independent PHASE audit and separate human-owned Maxwell/Bench commits precede
root/checkpoint recording once after both merges. Covered successful controls
are not rerun; frozen bytes are reused, with only four changed document whole
inputs saved once. Three unrelated original LIVE collectors remain untouched.
GSP-01/GSP09 and Goal remain OPEN/ACTIVE. All existing physical failures remain.
Full original physics/f30/all-N/crossover/complexity/high-frequency/general
material/periodic/nonlinear scope, whole host/device solver and setup/action/
backward peaks, current four-repository full acceptance and at least two
independent FINAL Goal audits remain OPEN with unchanged thresholds. No
production qualification or FINAL Goal audit follows.

### Original native radial-series M2L storage prerequisite completed

Three primary modules factor the original Helmholtz radial recurrence without
changing its arithmetic or the original full three-dimensional Taylor composer,
and append exact native radius catalogs, univariate owners and bounded actions.
Every physical displacement remains independently retained. Owner sharing uses
only identical original CUDA radius bits under the same scalar medium, order
and layout; views refuse any scalar-radius mismatch. No rounding, radius bins,
signed-axis projection, geometry change, approximation or order reduction.
Other393 Maxwell and all69 Bench primary modules remain exact. This internal
prerequisite does not yet change public FMM construction or capacity admission.

The unlaunched V1 independent audit found a whole-RHS finite mask outside the
admitted4-RHS action chunk. Every original V1 lexical input was closed before
correction in /tmp/gsp-radial-series-m2l-preaudit-failure-v1/manifest.json.
V2 checks finite values within admitted pair/RHS chunks before composing jets;
a new257-RHS mask-size control passes. No V1 numerical execution occurred.

Original V2 handle94250 drained actualrc0/24 NEW PASS,
XML9.401s, all423 lexical hashes exact. Original recorded
PIDs{'runner_pid': 2108218, 'pytest_pid': 2108637} are absent outside sandbox; LIVE observation and any
fast-process observation gap are recorded exactly in terminal-v1.json.
Three new radial p0/p4/p7 controls consume the existing original all48 native
coefficient/table artifacts once, without reference regeneration. Original
coefficients, F/H, native Hermitian and current/higher gradient errors stay
below1e-11 with zero saved forward tensors. Exact native states precede gates.
The NEW p12 nonorbit equal-radius control produces its original frozen-module
reference once, shares one owner across32 views, and rechunks requested B32
and RHS4 into admitted pair chunks without degree reduction. Its F/H errors
are {'F': 3.098330800094959e-17, 'H': 3.4949535320128143e-16}; actual retained inventory14816B vs
2113280B for32 original unshared compact owners.
That comparison covers retained native tensors only, not whole peaks/timing.

Order0..12, rows1..16, RHS1..4 per numerical chunk and requested pairs1..32
remain explicit. The visible128MiB action gate conservatively includes
temporary original Taylor composition before jets; final outputs, caller
state, allocator/library/host and whole setup/action/backward peaks are excluded.
Ten mutation refusals, retained/workspace preallocation refusal, seven invalid
factory inputs and the exact radius catalog boundary complete the24 controls.
No old planner tests, successful numerical controls, reference production,
producer/archive or full regression are replayed. Three unrelated original
LIVE jobs remain immutable; f30 minimum capacity PASS does not establish its
selected-order capacity or any action/solve/field/physical qualification.

Independent PHASE audit precedes a separate human-owned Maxwell commit;
root/checkpoint append once AFTER merge. GSP-01/GSP09 remain OPEN and Goal ACTIVE.
Existing failures, original final scope and thresholds remain unchanged.
Public integration, f30, high-frequency/changing carriers, physics, whole
resources, complexity, general material/nonlinear/periodic qualification,
current four-repository full acceptance and at least two independent FINAL
Goal audits remain OPEN. No production qualification or FINAL audit follows.

### Public original native radial-series M2L integration completed

Eight primary modules integrate the accepted original univariate radial owners
into public FMM configuration, exact radius catalogs, kernel/RWG order ledgers,
aggregate/component retained admission, M2L construction and regional actions.
Other388 Maxwell and all69 Bench primary remain byte exact. The accepted native
expansion/composition and bounded action primitive are unchanged. The native
radius-key loop is factored for bounded1024-node-pair metadata. Every original
physical vector remains independently retained, including24 native bytes per
view in ALL table/capacity/order ledgers. No radius rounding, bins, signed-axis
projection, geometry/kernel/order reduction, independent numerical alternative
or physical partition change. Native free-space complex128/order0..12/rows1..16
and requested pairs1..32 remain explicit; original full Taylor jets compose
only in admitted128MiB visible numerical chunks with exact degree rechunking.
Whole constructor displacements/host topology, final output/caller state and
allocator/library/Riesz/solver/setup/action/backward peaks are not qualified.

The explicit cartesian_fmm_radial_series_config_v1 fixes all current fields,
precedes near-cache version branches and refuses mismatched old storage/version
identities. Existing payload/construction identities stay exact. Two actual
source-preparation assertions stopped BEFORE provider writes; their whole
scripts and eight original source bindings are preserved. The unlaunched V1
preaudit P1 found an accidentally always-truthy tuple in old signed-axis restore.
All424 V1 lexical inputs were closed BEFORE the one-line scalar-comparison
correction; the new affected-old-payload control proves exact restoration.
No numerical execution occurred in any unlaunched failure.

Original public producer handle8170 drained actualrc1/
11 NEW PASS plus2 synthetic-order-fixture FAIL, XML162.813s, all
433 lexical hashes exact. The eight-point synthetic fixture
produced no original FAR pairs, and its independent oracle stacked an empty
list. Every433 actual epoch lexical input and all actual native/public states
were preserved before changing only the synthetic bookkeeping fixture to an
original support grid with actual FAR pairs. The original failed CUDA fixture
tree was not serialized; this gap is retained and no original tree bytes are
fabricated. New failed-only handle35142 drained actualrc1/
1 NEW PASS plus1 RWG-mock FAIL, XML5.464s, all443 lexical hashes exact.
All396 production modules and every other test function remain exact; only the
two failed controls run, saving their actual CUDA fixtures before planning.
The kernel ledger passes; the RWG mock returns an incorrect Python scalar,
where original logaddexp requires a native CUDA tensor. Original handle
35142 actually returned1/1PASS+1FAIL and that full failed
input plus both actual CUDA fixtures is closed before correcting ONLY the
RWG mock scalar type. Final failed-only handle35049
drained actualrc0/1PASS, XML6.651s, 453 exact
lexical hashes; the successful kernel ledger is not replayed. The13 distinct
new controls pass in COMPOSED epochs11+1+1, not one all-green run.
The successful public native/gradient/solve/restore control and reference are
NOT replayed.
Original PIDs{'runner_pid': 2574663, 'pytest_pid': 2580749} and actual LIVE/terminal absence evidence
are recorded in terminal-v1.json. The new public storage policy independently
declares the original143MHz four-lossy-magnetic-body/72-current Scene and fixed
original admittedp7 before loading the ONE closed original72x72 weak reference
SHA837bcf6cf2f3870c11fee5926d840f0744aec3403a4c06ce17ded28a21aa2e2e.
No originalp6..12 search, reference production, old planner test or successful
producer is replayed. Native public errors are {'columns': 4.1259403542484797e-13, 'forward': 4.118100964466714e-13, 'original_H': 4.141840377783028e-13, 'native_H': 1.8132401671065414e-16, 'current_gradient': 1.7471722882973528e-17, 'higher_current_gradient': 5.35687522056158e-16}, below original
1e-8 coefficient/F/H and1e-11 native Hermitian/current/higher gates, with zero
saved forward tensors. Every actual native/solver state precedes failure gates.
Actual radial owners3 cover every
12 physical FAR view, retaining
12792B M2L; complete retained native inventory
1303288B equals the original structural plan exactly.
Original near cold149 calls and warm0 across actions
and solve; no extra warm integration. Original GMRES/error admission passes,
rtol1e-12/maxiter240/restart120/no fallback unchanged. Complete original weak
residual4.747810795463683e-12 <=1e-10. A geometry-bound17-RHS M2L
control confirms finite masks remain within accepted chunks. One public exact
restore succeeds with construction/solve banned; changed storage identity refuses.
One new Result SHA6bff5d70039562592cf496143050e47297d6191c268edde0740be8d74a92984c is produced.

The other new controls cover physical node-pair catalog1057/one-ULP/nonorbit,
p0/p3/p12 retained accounting with independent CUDA radius bits and actual
native owners, complete original physical partition coverage, five pre-moment
component budget refusals, and both kernel/RWG order ledgers. Those two ledgers
use explicit MOCK bounds to verify new storage arithmetic without replaying old
kernel contours; they do NOT establish automatic-order physical qualification.

Original Bench consumer handle55927 drained actualrc0/
1 NEW PASS, XML7.325s, all496 lexical hashes exact.
Physical/numerical declarations precede producer hash/open. ONE archive write/
read preserves native schema11, exact J/M and the new storage/cache identities;
14 changed-configuration refusals and construction/solve bans pass. All10 actual
archive arrays are verified against manifest SHA. No reference, producer,
construction, solve or old successful archive is replayed. Actual PID observation
gaps, if any, remain explicitly recorded; no LIVE evidence is invented.

Independent PHASE audit precedes separate human Maxwell/Bench commits and ONE
root/checkpoint append AFTER both merges. Only four changed frozen document
whole inputs are saved once; all other immutable bytes are reused with explicit
lexical epoch bindings rather than repetitive archives. Three unrelated
original LIVE collectors remain untouched. GSP01/GSP09 OPEN and Goal ACTIVE;
all existing failures, original scope and acceptance thresholds unchanged.
No selected-order f30, general-material/nonlinear/periodic, high-frequency/
changing-carrier, physical, whole-resource, allN/crossover/complexity/production
qualification or FINAL Goal audit follows. Current four-repository full
acceptance and at least two independent FINAL Goal audits remain OPEN.

### Original-scale radial-series resource workflow completed

The primary resource runner and all-point summarizer now accept a versioned,
SHA-bound bridge from the closed original 24-case resource cohort to original
native radial-series M2L storage. All396 production modules remain unchanged.
The only representation change is M2L; original geometry, support-box midpoint
centers, near full coefficients, quadrature, leaf/partition, rows/RHS, retained
budgets, kernel/native gates, all four N and both frequency families remain.
Fixed original orders6/6/6/11 and growing6/6/6/12 are reused with their OLD
source identities separate from current source hashes. Every original frozen
dependency and all24 records are verified. Current constructors independently
admit native couplings and capacity; successful original searches are not replayed.

Seventeen NEW targeted controls pass:13 closed-receipt metadata guards and4
tiny CUDA final-capture/failure-state guards. They run no original FMM/reference/
integration/planner/archive controls. The unlaunched v1 audit P2 was preserved
with all859 actual whole lexical inputs BEFORE correction. Failed final native
gates now capture the last allocator peak and save actual x/y/F/H/gradient and
phase metadata after the measured interval, then raise unchanged failures.
Successful vectors are not archived. No resource cohort is yet claimed closed.

The original cohort stored weak error reports without matrix bytes. New radial
cohort accuracy remains null; the summarizer rejects relabeling historical errors
as new measurements. Current public all-column radial controls remain separate.
Complete original24 new resource measurements are the next dependency, not a
partial/tail-only fit or reduced N. Shared GPU timing interference, whole solver/
host resources, all-size accuracy, dense/H2 crossover and asymptotic complexity
remain OPEN. No new resource/physics/production qualification or FINAL Goal audit.
OriginalLIVE f30/tensorAC/magneticDAG providers remain immutable, all failures
preserved, GSP01 OPEN/Goal ACTIVE, original complete scope and thresholds
unchanged. Current four-repository full acceptance and at least two independent
FINAL Goal audits remain required. Independent phase audit precedes this separate
human stage commit; root/checkpoint append ONCE after merge.
