# RFIC MPIE MoM R0 规格冻结

> 状态：冻结，2026-11-06。用户指示由 RFIC 负责人 完整推进本线（"请你先做完整推进"），R0 决策按推荐项冻结，不单独等待签字。之后改动这些决策要在本文件记一条修订，并给出原因
> 上级计划：`rfic_mom_development_plan.md`
> 分支：`sie/rfic-mom`（worktree `<workspace>/wt/sie-rfic-mom`），证据目录 `<evidence-root>/rfic-mom-20261106/`
> 约束：不改动 FDTD 代码路径（`fdtd/`、`ports.py`、`stepping.py`、编译器）。GPU 使用遵守协调者 2026-11-06 的分配规则（见第 9 节）。

## 1. 约定

- 相量约定为 `exp(-iωt)`，与现有 MoM 一致（`single_layer.py`：`G = exp(ikR)/(4πR)`）。文献里的 `e^{jωt}` 公式一律做 `j → -i` 变换，变换后在代码注释里写一次结论。
- 单位用 SI，z 轴向上，层从下往上编号。
- 径向谱变量为 `kρ`，每层的纵向波数 `kz_i = sqrt(k_i² - kρ²)`，取 `Im kz_i ≥ 0` 这一支（辐射条件）。

## 2. 工艺栈 schema

新增公共类型 `LayerStack`，放在新模块 `witwin/maxwell/layers.py`：

```python
mw.LayerStack(
    layers=(mw.Layer(name="sub", thickness=300e-6, eps_r=11.9, sigma=10.0),   # S/m
            mw.Layer(name="ox",  thickness=10e-6,  eps_r=4.1),
            ...),
    bottom="pec" | mw.HalfSpace(eps_r=..., sigma=..., mu_r=...),
    top=mw.HalfSpace(eps_r=1.0),          # 或 "pec"
)
```

- 每层的参数是 `eps_r`（复数或实数）、`sigma`、`mu_r`（v1 只支持 1）。损耗按 `eps = eps_r + i·sigma/(ω ε0)` 计入，只计一次，规则和 `physics.py` 相同。
- `Scene(background=LayerStack(...))`：`scene.py` 的 background 类型检查要放行 `LayerStack`。这一处在 R2 公共接入时改，改动不超过 3 行。FDTD 准备阶段已经拒绝非空的 background（`scene.py:1917`），所以 FDTD 行为不变。
- 金属：
  - v1 是零厚度片（`Box` 的 z 向厚度为 0），每片必须落在某个 z 平面上，这个平面叫"金属平面"；
  - R4 起金属可以有厚度，用"上下表面 + 侧壁"建模；
  - 过孔（R3 起）是连接两个金属平面的竖直柱体或盒子。

## 3. Green 函数（R1）

- Formulation 用 Michalski–Zheng 的 C 型 MPIE（参考 Michalski & Mosig 1997, IEEE TAP 45(3)）。R1 只交付水平电流需要的两个分量：`G^A_xx`（等于 `G^A_yy`）和 `K^Φ`。过孔需要的 `G^A_zx`、`G^A_xz`、`G^A_zz` 以及 C 型修正项放到 R3。
- **谱域：** 用 TL 等效电路求每个 kρ 的电压和电流 `V_i, I_i, V_v, I_v`，TE 和 TM 两种模式，然后组合成 `G̃^A_xx(kρ; z, z′)` 和 `K̃^Φ(kρ; z, z′)`。
  - CUDA kernel K1 每个线程算一个 (kρ, 平面对, 频率) 节点，在线程内按层递推，全程 fp64。
- **2.5D 简化：** v1 的源和观察点都在离散的金属平面上，所以每个 (z_i, z_j) 平面对的 Green 函数只依赖 ρ，是一维表。
- **Sommerfeld 变换：**
  `G(ρ) = (1/2π)∫_0^∞ G̃(kρ) J_0(kρ ρ) kρ dkρ`（K^Φ 和 G^A_xx 用的都是 J_0）。
  - 积分路径先变形，绕开靠近实轴的极点和支点，在 `exp(-iωt)` 下由实现者推导偏离方向，再由 oracle 用另一条路径验证；然后沿实轴积到 `t_max`。
  - 先减去 `kρ → ∞` 的渐近项（准静态项），使余项快速衰减。截断误差有估计并写进报告。
  - **实现方式：** K1 在路径节点上生成谱样本 `Gs[q, m]`（m 是平面对、分量、频率的组合）。K2 生成 Bessel 矩阵 `B[r, q] = w_q J_0(kρ_q ρ_r) kρ_q`，每个频率批次一份。然后调 cuBLAS ZGEMM 算出 `T = B · Gs`。这一步不写 Python 循环，也不自己写 GEMM。
- **准静态提取：** 渐近项写成 `1/R` 型项（直接项加最近界面的镜像项，`R = sqrt(ρ² + Δz²)`），系数取 kρ→∞ 极限下的反射系数，解析求值。表里只存余项 `G_s(ρ) = G − G_qs`。
- **表：**
  - ρ 网格在 `[0, ρ_max]` 上分两段，近段均匀、远段对数间距，ρ_max 是版图对角线的 1.05 倍；
  - 插值用三次 Hermite 或单调三次，fp64 存储；
  - 表的布局是 `[频率, 平面对, 分量, ρ]`，内存连续；
  - 准入条件：在网格中点处，插值和直接求和的相对误差 ≤ 1e-6。超出准入范围（ρ > ρ_max、或 z 不在任何金属平面上）就报错。

### R3a: vertical formulation-C derivation (exp(-i omega t))

Use q=k_rho, k0=omega/c0, e_s=eps(z'), e_o=eps(z),
h_s^2=k0^2 e_s-q^2 and h_o^2=k0^2 e_o-q^2. H and E denote the
normalized TE and TM shunt-source voltages V_i/(i omega mu0); their scaled
impedances are 1/h and h/(k0^2 eps). D=k0^2(H-E)/q^2 is the existing
horizontal scalar spectrum, also for points inside a layer.

For z'<z, analytic outgoing-solution logarithmic derivatives are

    L_s = -i h_s (1-r_minus(z'))/(1+r_minus(z'))
    L_o = +i h_o (1-r_plus(z))/(1+r_plus(z))
    d_s V = L_s V; d_o V = L_o V; d_os V = L_s L_o V.

Here r_plus/minus are reflections referred to the evaluation coordinate.
These follow by differentiating the incident/reflected exponentials. The
coincident limit takes opposite one-sided derivatives, excluding the electric
dyadic contact delta. Native evaluation factors V=K*(1+r_minus(source))*
(1+r_plus(observer)) and differentiates the factors directly. This cancels
the apparent logarithmic-derivative poles at PEC endpoints analytically,
rather than evaluating zero times infinity. Modes algebra carries the scaled TE-TM differences
through the derivatives. The time-convention conversion is j -> -i from
Michalski and Zheng (1990) and Michalski and Mosig (1997), sections IV-VI.

The corrected formulation-C spectra are

    A_zz   = d_os D/k0^2 + q^2 d_os E/(h_s^2 h_o^2)
    A_zrho = q (d_o D/k0^2 - d_o E/h_o^2)
    A_rhoz = -q (d_s D/k0^2 - d_s E/h_s^2)
    K_phi  = D.

The inverse transforms use q J0(q rho)/(2 pi) for zz/phi and
q J1(q rho)/(2 pi) for the cross components. Cartesian x/y cross terms
multiply the radial coefficients by dx/rho or dy/rho; on-axis projections vanish.
Explicitly, C_phi=d_s D-k0^2 d_s E/h_s^2 and
U_e=k0^4 e_s e_o d_os E/(h_s^2 h_o^2), the normalized dual series-source
current, give A_rhoz=-q C_phi/k0^2 and
A_zz=U_e/(k0^2 e_s)+d_o C_phi/k0^2.
Thus the vertical scalar correction is already grouped into the vector terms:
assembly uses one scalar D and must not add C_phi again.

For |q^2|>4 max(|k0^2 e_s|,|k0^2 e_o|), use the equivalent stable expressions

    A_zz = d_os H/q^2
         + d_os E*((k0^2 e_s+k0^2 e_o)*q^2-k0^4 e_s e_o)/(q^2 h_s^2 h_o^2)
    A_zrho = (d_o H-k0^2 e_o d_o E/h_o^2)/q,

and the negative source-derivative counterpart for A_rhoz. Expanding the
numerator analytically avoids subtracting O(q/k0^2) terms in the zz static
limit. At small q the carried D form instead avoids TE-TM cancellation.
Canonical ordering gives A_zz(z,z')=A_zz(z',z), K_phi(z,z')=K_phi(z',z),
and A_zrho(z,z')=-A_rhoz(z',z) with observer-minus-source radial directions.

For a static ray of distance d=alpha z'+beta z+gamma with TE coefficient B
and scalar coefficient P, both multiplying exp(-qd)/(2q), the coefficients are

    P_zz=alpha*beta*(B-(e_s+e_o)*P)
    P_zrho=beta*(e_o*P-B); P_rhoz=-alpha*(e_s*P-B); P_phi=P.

Their inverses are 1/(4 pi R) for J0 and rho/(4 pi R(R+d)) for J1,
R=sqrt(rho^2+d^2). The local direct wave is extracted exactly as
exp(i k R)/(4 pi R) for zz and that wave divided by eps for phi.
PEC-only rays remain in one homogeneous region and use the same full-wave
extraction with their signed image weights. This removes the reflected-wave
corner at a grounded span endpoint; other rays retain static extraction.
Homogeneous cross terms vanish. A PEC ray has B=-1, P=-1/eps and
alpha*beta=+1, so its vertical coefficient is +1; horizontal images stay negative.

Additional table storage is [frequency, reciprocal z-node pair, 4, rho],
ordered (zz,zrho,rhoz,phi). Horizontal arrays and evaluation arithmetic remain
unchanged. Via spans are merged and split at interfaces, with eight interior
Chebyshev-Gauss nodes per segment by default (z_nodes controls degree).
Interpolate the smooth remainder polynomially in both z coordinates and
reconstruct each analytic image at its actual affine distance. Interior
interfaces select the upper segment; outer endpoints use the available
one-sided segment. The ray list is fixed per segment pair over its extraction
budget. Deformed rho bins, vendor complex128 GEMMs and decaying Hankel tails
are retained for J0 and J1. Truncation additionally resolves z-node distances
to interfaces; independent chunk budgets bound spectral and Bessel storage.

Rho admission checks every coarse radial midpoint at every z-node pair.
Z admission directly transforms tensor products of every inter-node midpoint
and both one-sided segment endpoints at 17 positive radial locations across
the grid. Both reject error above 1e-6 of the maximum total component on
each checked radial row. Rho admission floors structural cross components
at 1e-6 times the zz row maximum; z admission applies this floor to any
vanishing component, including the scalar kernel at a PEC endpoint. Coincident rho=0 singularities stay analytic and are
excluded from admission samples. These are a posteriori checkpoint tests,
not rigorous uniform interval bounds. Out-of-span z, out-of-range rho and
absent declared via spans fail closed.

## 4. MPIE 装配（R2）

- **Galerkin 形式：**
  `Z_mn = -iω μ0 ⟨f_m, G^A_xx f_n⟩ + (1/(-iω ε0)) ⟨∇·f_m, K^Φ ∇′·f_n⟩`
  - 符号和归一化要做到：均匀栈极限下与 `single_layer.py` 的 `iωμ(vector − divᵀ scalar div / k²)` 一致，作为门 G-R2-1。
- **积分：**
  - 准静态 `1/R` 项：内层积分用解析公式（标量位 `∫1/R`、矢量位 `∫(r′−v)/R`，对任意观察点都成立，包括不共面的近距离点），外层用 Gauss 积分。
  - 余项 `G_s`：在 Gauss 点上从表插值，内外层都用 Gauss 积分。
  - 近和远按距离与三角形尺寸之比分类，分类结果是离散量。
- **确定性：**
  - 按源三角形块划分 tile，tile 的顺序固定。
  - 每个 tile 先把面对的 3×3 矢量块和 P0 标量值写进 tile 缓冲，再由 gather kernel 用一个线程负责一个 (m, n)，按 (T±, S±) 的固定顺序累加。
  - 不用浮点 atomic。重复运行必须 `torch.equal`，跑 3 次。
- **精度：**
  - 近场类和解析部分用 fp64。
  - 远场类的算术用模板参数选精度，默认值由 R2 实测决定：fp32 对 S 参数的影响 ≤ 1e-6 就用 fp32。
  - Z 以 complex128 保存。

## 5. 求解（R2 dense；R8 迭代）

- dense 求解：
  - 用 complex64 LU（cuSOLVER，通过 `torch.linalg.lu_factor`）加 fp64 迭代精化：每轮算 complex128 残差 `r = b − Z x`，用 complex64 求修正量；
  - 收敛条件 `‖r‖/‖b‖ ≤ 1e-10`，精化最多 10 轮，不收敛就报错；
  - 所有端口的 RHS 一次批量求解。
- 显存超出 dense 能力时拒绝运行，提示改用 R8 的迭代后端。

## 6. 端口和网络（R2）

- 复用 `mw.LumpedPort` 的名字、positive/negative 两个点和参考阻抗，这样同一个 Scene 也能在 FDTD 上跑交叉验证。
- MoM 的端口模型：在 positive 和 negative 两点之间放一条桥接金属条。条宽由 `MoMConfig` 的端口选项显式给出，没有默认值。条中点的切线上放 delta-gap 电压源。
  - 两个点都必须落在同一金属平面的金属上；不满足就报错。
  - 竖直端口放到 R3。
- 网络计算：依次在每个端口加 1 V 激励，得到所有端口的电流，构成 Y 矩阵。再换算成 Z 和 S，输出现有的 `NetworkData`（`network.py:596`），用 `write_touchstone` 写文件。
- 去嵌入：v1 不做，桥接条的寄生电感会记录在元数据里。R7 用校准结构法做去嵌入。

## 7. CUDA 工程约定

- 新扩展名为 `witwin_maxwell_mom_layered`，源码放 `witwin/maxwell/mom/layered/cuda/`，构建脚本 `witwin/maxwell/mom/layered/build.py`。
  - 复用 `fdtd/cuda/build.py` 的 `_cuda_gencode_flags` 和 `_conda_torch_ldflags`，nvcc 参数加 `-O3 --fmad=false`。
  - ABI 版本标记格式为 `mom_layered_abi_v1`。
  - 构建目录环境变量是 `WITWIN_MAXWELL_MOM_LAYERED_BUILD_DIR`，默认 `<scratch>/witwin-build-rfic-mom`。
- 所有 kernel 都以 torch op 的形式暴露，输入输出是 CUDA tensor，host 端只传形状和标量。
- **性能证据（替代 Nsight Compute）：** 这台工作站不能跑 ncu（`RmProfilingAdminOnly=1`，没装 ncu，11-04 已决定不改驱动）。每个 kernel 的性能证据改为：
  1. 解析 bytes/FLOP 模型，写在 kernel 头注释里；
  2. 实测峰值：fp64/fp32 FMA microbenchmark 和 device-to-device copy 带宽，每次计时前在 GPU 1 上测一次；
  3. 实测 kernel 时间，换算成模型给出的有效带宽或 FLOP 率，和实测峰值的比值 ≥ 50%（主导资源）。

  ncu 抽查需要云端机器；协调者规定本工作不使用云 GPU，所以暂不做。
- 正确性测试在 GPU 0 上跑（受 memcap 约束），计时在 GPU 1 上跑（在 gpu1.lock 下，每次 ≤ 30 min，事先通知协调者）。

## 8. Oracle 和外部参考

| ID | 内容 | 独立性 |
|---|---|---|
| O1 | 谱域 TL：CPU numpy fp64，用阻抗传输公式（`Z_in` 递推），不用 kernel 的反射系数递推 | 公式不同 |
| O2 | Sommerfeld：mpmath/scipy 自适应积分，路径与 kernel 不同（实轴加极点附近的半圆绕行，极点由求根得到），再检查和 kernel 结果的路径无关性 | 路径和数值方法都不同 |
| O3 | 解析极限：均匀栈（Sommerfeld 恒等式 → `exp(ikR)/(4πR)`）、PEC 地（精确镜像）、介质半空间的低频准静态镜像 | 解析解 |
| O4 | 装配：CPU fp64 高阶暴力积分（小网格）；均匀栈极限对比现有 `single_layer.py` 和 `quadrature.static_potential_integrals` | 现有独立实现 |
| X1 | **开源 RWG MoM（scuff-em：`scuff-rf` 加 `libSubstrate` 隐式多层衬底）**：多端口 Z/S 参数，带衬底和地平面；自带微带、CPW、贴片的例子。R2 起作为**首选外部参考**。源码构建，只在 CPU 上运行 | 外部、同类方法（MoM）、独立实现 |
| X2 | **开源分层介质 Green 函数库（Strata）**：dyadic MGF，支持 MPIE 各分量。R1 的外部 oracle，与 O1–O3 并用 | 外部、独立实现 |
| I1 | 仓库内的微带和耦合线闭式公式：`tests/support/coupled_line_references.py`（Hammerstad–Jensen、Kirschning–Jansen） | 解析解 |
| I2 | 仓库内的 2D 截面线路求解器（准静态和全波，支持 Z_s）：`benchmark/line_cross_section_reference.py` | 方法不同 |
| I3 | 本仓库 FDTD PCB 线（同一个 Scene，同一套 `LumpedPort`）：只读调用，不修改 FDTD 代码 | 方法不同 |
| E1 | 开源 FDTD，从源码构建。现在装在易失的 `/tmp/pcb-reference/install` 里，R2 前迁到 `<evidence-root>/rfic-mom-20261106/tools/`；驱动在 `scripts/dev/fdtd/*_external_reference.py`；已有微带、弯折、差分对的缓存 | 外部、方法不同 |
| E2 | 开源 PEEC 磁准静态电感提取工具（FastHenry2，源码构建，CPU）：无衬底时电感的 L(f)、R(f)。本机没有，R4 前构建 | 外部、方法不同 |
| E3 | 开源全波 FEM（Palace，docker 容器，CPU）：带衬底时的 S 参数，lumped port。本机没有，R2 后构建 | 外部、方法不同 |
| E4 | 仓库现有的第三方云 FDTD adapter：adapter 还不能导出 lumped/terminal port，最近记录的余额约 19.4 单位。每次估价并报用户批准；只在 E1–E3 不够时才用 | 外部 |
| E5 | 用户手里的商业工具许可或测量数据：待用户提供，本仓库文档不写产品名 | 外部 |

外部参考盘点见 协调者 的调查记录：`<evidence-root>/rfic-mom-20261106/external-reference-inventory-20261106.md`，含产品名，不进仓库。结论：本机没有分层介质 MoM，没有电感或变压器的测量数据，也没有 Sommerfeld 代码。所以 R1 的 Green 函数只能靠 O1–O3；过孔电感靠 E1、E3 和解析近似。

外部参考是 M2/M4 的首选参考（D3）。用户 2026-11-06 补充："开源solver能不能使用参考 MoM的最好"。因此优先级是：X1/X2（开源 MoM 和 Green 函数），然后 E1–E3，最后 E4/E5。X1、X2、E1–E3 都只在 CPU 上运行（nice，8 线程），不使用 GPU。源码和构建产物放在 `<evidence-root>/rfic-mom-20261106/tools/`，独立的工具链环境是 `tools/env`，不改动 maxwell 环境。

## 9. 资源规则（协调者 2026-11-06 的分配）

- GPU 0：只做正确性工作，受 memcap 约束；空闲量 ≥ request + 10 GiB 才启动；LQ-3 进程不能动。
- GPU 1：只做计时，必须在 `flock <evidence-root>/coordinator-20261106/gpu1.lock` 下运行，安排在 core-CUDA 的持有间隙，每次 ≤ 30 min，事先通知协调者。
- 不使用云 GPU。CPU 任务用 nice -n 10，线程数 8。

## 10. 修订记录

- 2026-11-11: R6-3 passive wideband macromodel (C3, macromodel.py) + wideband sweep (D3). Fit: the shared
  vector-fitting algorithm on S (initial poles, real basis, relocation, stable canonical poles reused from
  rational.py) with Householder-QR least squares; the shared fit's default CPU lstsq driver (gelsy) varies at
  1e-16 between calls, QR is bitwise repeatable. Certificate: bounded-real Hamiltonian M (R = D^T D - I,
  Q = D D^T - I); imaginary eigenvalues = singular-value crossings of 1; a stable model with |D|_2 < 1 and no
  crossing at f > 0 is passive from DC to infinity. Eigenvalues within 1e-6 of |lambda|_max are the DC point
  (sigma(S(0)) = 1 is the lossless DC limit of an open even mode, allowed); unconfirmed near-axis eigenvalues
  void the certificate. Enforcement: clip sigma(D) to 1 - margin, then minimise sum_f |dC X(s_f)|_F^2 (sampled
  Gramian over the samples + a 512-point band grid) subject to sigma_i + Re(u_i^H dC X v_i) <= 1 - margin at 17
  points per violation band (accumulated), exact QP in whitened coordinates via its dual (Lawson-Hanson; a primal
  active set cycled on the loop fixture), repeat until the Hamiltonian test passes. A clipped D is followed by a
  least-squares refit of C to the samples (a 3e-3 D clip otherwise shifted the whole loop response).
  AdaptiveSweep: a surrogate failing its in-band check is enforced the same way if the change is <= tolerance/4
  (enforcement_change), else rejected as before; passing sweeps unchanged (F-A test_layered_adaptive 30/30). Bound: max_error <= fit_max_error + enforcement_change (triangle inequality).
  Export: exact LFT to the SI Y realization (A - B(I+D)^-1 C, ...), SPICE subcircuit of 1 pF state capacitors
  and VCCS cards; read back by an independent nodal analysis (1e-12). Public mw.mom.passive_macromodel,
  PassiveMacromodel, MacromodelReport, HamiltonianReport (experimental mom_layered); exported in mom/round7
  with its conventional consumer re-receipt (mw.mom/__init__.py is a pinned consumer-epoch file). Streamed mm-wave
  calls take AdaptiveSweep: each sample is its own single-frequency streamed solve and merge_streamed_sweep
  regroups the frequency-sorted samples by wall mesh (guard 5e7a41c3 retired; gradients keep a new
  gradient-only guard). Matrix_free/hmatrix multi-frequency calls warm-start block GMRES from the previous
  frequency's loop/tree coefficients (single-frequency calls and the first frequency unchanged bit for bit;
  later frequencies move within the 1e-5 admittance stopping bound). Evidence <evidence-root>/rfic-mom-20261106/r6-3.
- 2026-11-11: R4-2 per-frequency mm-wave Green tables (frequency_groups.py). A graded vertical table
  stores every frequency at the graded size (0.5-1.8 GB per frequency, 6.1 GiB build peak per via-stack
  frequency), so a 4-frequency 60-200 GHz call ran out of memory at 7.5 GiB. Now the full-set
  ungraded step 0 runs exactly as before; if it admits, the one-table path is unchanged (A/B vs
  e42174e6e on ungraded 1/3/5 GHz microstrip, 1/10 GHz and 5 GHz via-stack networks: S, z0 and
  currents torch.equal). If it fails with dielectric interfaces and >1 frequency, the table build
  returns None (vertical_refine defer_grading) and prepare returns a StreamedLayeredNetwork; the
  solve runs the exact single-frequency prepare+solve per frequency (own table, level, wall mesh),
  frees it, and concatenates S/z0 exactly. Each frequency is torch.equal to its single-frequency
  call by construction; peak = largest single-frequency peak. Rejected: one multi-frequency graded
  table per level group -- table bits depend on the band (k_max, truncation, radial grid, GEMM
  chunking use fmin/fmax), so it cannot equal per-frequency calls, and memory-sized groups would
  make bits depend on free memory. Groups = maximal runs of consecutive frequencies on one wall
  mesh; currents per group (GroupedLayeredSurfaceData.groups); metadata frequency_streaming,
  frequency_groups, frequency_metadata, per-frequency tuples concatenated. Sweep/gradients and
  single-mesh currents/layout fail closed (3 census guards). Predictor: the first level no longer
  credits (K0/12)^4 (via stack K0 = 8: ungraded -> level-1 drop measured 9x, not 81x); K0 = 12
  unchanged. Gates (GPU 0, 60/100/150/200 GHz, one call): microstrip levels 2/2/3/3, 2 groups,
  5.33 GiB total device peak at a 7.5 GiB torch cap; via stack levels 1/2/2/2, 2 groups, 6.53 GiB
  total device peak at a 7.1 GiB torch cap (at a 7.5 GiB torch cap the caching allocator reserved
  7.4 GiB + context = 7.7 GiB, so the slot run uses 7.1); every frequency torch.equal to its
  single-frequency call (S, z0, currents, mesh). Evidence <evidence-root>/rfic-mom-20261106/r4-mmfreq.
- 2026-11-11: round 3 (G1-G4 merged on mom/round3) + horizontal admission family floor (interp.cu
  admission_kernel). The {0 PEC, 100, 110} um via-stack horizontal table at 1 GHz read 6.08e4: the
  (0 PEC, 110 um) tangential row cancels against its image to ~1e-14 and was normalized by its own
  row maximum. Each row's denominator floor is now 1e-6 x the maximum row scale over all plane
  pairs of the same frequency and component (the C1 rule of vertical.cu horizontal_admission);
  fixed-order per-thread maxima, no atomics. That table now admits at 6.5e-10 with a bitwise
  repeat. Proof on 12 previously admitted tables (G1's 11 identity cases + the R6b fixture table):
  all 30 tensors bitwise equal before/after; the admission scalar is unchanged except the ground
  case 9.3e-15 -> 2.0e-15 (verdict only). Evidence <evidence-root>/rfic-mom-20261106/round3/probes.
- 2026-11-10: G1 mm-wave vertical Green-table z admission (graded z segments, vertical_refine.py).
  Cause: the vertical remainder is the total kernel minus static (q -> inf) images. At a
  dielectric interface the next reflection term, O(k^2/q^2) exp(-q d), leaves k^2 d log d in the
  image distance d = |2 z_i - z - z'| (and in the transmitted distance across the interface), so
  one Chebyshev polynomial per layer converges like f^2 K^-4 (measured on the 254 um eps 9.8
  grounded microstrip, K = 8..24, 30..200 GHz; the worst rows have z' on the substrate/air
  interface and rho -> 0). PEC images are exact and do not contribute.
  Algorithm (one adaptive loop, coordinator-accepted 2026-11-10): step 0 builds and measures the
  ungraded table exactly as before; e0 <= 1e-6 returns it unchanged (bitwise; an a-priori
  predictor cannot promise this). Otherwise the table is rebuilt with 12 nodes per piece and
  geometric pieces of ratio 4 toward every dielectric segment end (cuts at L*4^-i, i = 1..levels,
  L the segment length, L/2 when both ends are graded). Corner error ~ delta^2 K^-4, so the first
  level is ceil(log16(e0 (K0/12)^4 / 1e-6)); every failure adds one level; level 5 raises
  ConvergenceError (census safety guard). Graded tables use a truncation per z pair,
  min(T_ungraded, max(T_horizontal, 64/(eta_s + eta_o))) in power-of-two buckets: every
  non-exact remainder path touches an interface, so its spectrum decays like
  exp(-q (eta_s + eta_o)); the ungraded rule 32/min eta is the maximum of this bound over pairs.
  Wall meshes take the piece boundaries as z breakpoints (assembly integrates each wall
  triangle in one segment). table.vertical.grading_record and network metadata
  "vertical_z_grading" (graded tables only) record levels, predicted levels, pieces and errors.
  Gates: table A/B of 11 previously admitted stacks vs fe59b3cd2 bitwise (30 tensors each);
  grounded microstrip admits at 60/100/150/200 GHz (z 1.6e-7/5.0e-7/6.9e-8/1.4e-7, levels 2/2/3/3,
  each the predicted level), bitwise repeat, network Y vs a one-level-finer table on the same
  mesh <= 2.1e-11. Via stack (PEC / 100 um eps 12.9 / 10 um eps 3 / air, underpass with two vias,
  horizontal ports; the 10 um span is graded from both ends): admits at 60/100/150/200 GHz with
  levels 1/2/2/2 (predicted 1: one loop step), z 9.9e-7/6.1e-7/1.8e-7/9.2e-7, bitwise repeat,
  network Y vs a level-3 table on the same mesh 8.4e-9/5.1e-11/8.4e-12/2.8e-11 (gpu0 runner,
  head peak 6.1 GiB, reference 11.7 GiB; the via-stack gate is @slow and needs a >= 7 GiB cap).
  Cost: step 0 2-3 s; graded vertical build
  10/17/34 s, peak 1.0/2.2/3.9 GiB, stored 0.47/1.04/1.84 GB per frequency at levels 1/2/3
  (ungraded K=12: 3 s, 0.48 GiB, 0.12 GB). A multi-frequency table stores every frequency: four
  frequencies at 60-200 GHz exceed a 7.5 GiB cap (open: per-frequency-group tables).

- 2026-11-10: G4 combination gates and lifted round-2 refusals (tests/mom/layered/test_layered_combinations.py).
  Polygon layouts x dispersive layers and surface_admittance x dispersive layers match per-frequency runs of
  the frozen permittivity eps(f) (E1 method; polygon <= 2.7e-12, surface_admittance <= 1.4e-9 over the low,
  blend and DSA regimes). GR1 gradients pass the GR1 central-FD bound on the planar mesher (forced triangle
  mesher and Prism vias; conductivity, roughness, thickness, via conductivity/roughness; errors <= 1e-6 of
  max|dY|) and agree with dense through hmatrix currents. Lifted: polygon x hmatrix (partition and near pairs
  in leaves on planar meshes with walls; hmatrix vs dense 1e-8..1.3e-7, exact check <= 1.4e-8), polygon x
  edge ports (planar._planar_edge_span: the C1 rules on the shapely metal union; edge == delta gap of the
  same width bitwise), surface_admittance x edge ports (aperture-wide flush PEC block; edge-fed split strip
  DC R = l/(sigma A) to 5e-10; bitwise the same-width delta-gap block). Defect fixed: the D2 body
  preconditioner keyed on layout.face_owners, which planar layouts also carry; it now keys on face_kinds
  (closed body meshes), since widened via-wall patterns were quadratic in the wall faces.
- 2026-11-09: F-B four-port transformer fixture and independent reference workflow.
  Two interleaved Manhattan two-turn windings use 20-um traces, 5-um spacing,
  360/310-um primary/secondary outer dimensions, centered 2/1-um top/underpass
  metal at z=305/302 um, four 4x4x3-um signal vias and a 30-um ground ring
  with inner/outer sides 430/490 um. The wider traces place the F-A-density
  fine sheet mesh near the requested 15k scale: max_edge/edge_cell=10/4.9 um,
  14411 unknowns and 10400 faces; coarse 80/20 um gives 1708 unknowns.
  Layout coordinates use an exact binary lattice with nominal 1-um pitch
  error below 3e-19 m, avoiding one-ULP duplicate bridge/rim breakpoints.
  The process stack and matching lower half-space are identical to F-A.
  Four 50-ohm single-ended ports join x=205..215 um at y=170/70/145/45 um:
  p1-p2 primary, p3-p4 secondary, both with the same circulation.
  With D=((1,0),(-1,0),(0,1),(0,-1)), Zw=D.T Z D describes zero common-mode
  winding currents; L=-Im(Zw)/omega, R=Re(Zw), M=L12 and k=M/sqrt(L1 L2).
  Reference source bridges retain their explicit cut/gap convention without
  deembedding. The quasi-static inductance reference uses the compiled metal
  union and direct admittance columns p1/p3 from four external terminal pairs,
  resolving finite copper cross sections and a 0.1-um thin-sheet control.
  Direct overlap and order-6/order-8 controls quantify the accelerated operator;
  a measured segment-preconditioner control preserves the printed L/M/k/R values.
  All ten quasi-static cases have rc=0. At 1 GHz the nominal 2.5-um case gives
  L1/L2/M=1.731400/1.457585/1.042330 nH and k=0.656130. Its 5-to-2.5-um
  self-L spread is <=0.611%, with resistance spread <=3.32%; the thin-metal
  self-L spread is <=0.219% over 1 MHz, 0.1 and 1 GHz. Order-6 to order-8
  changes L/M by <=0.00428% at those frequencies. These are finite spreads.
  The open-source FDTD reference uses four independent drives at 10/5/2.5-um
  lateral resolution, plus one p1 loaded-response boundary/z/source-gap control
  each at 5 um. Field acquisition uses nice 10, at most two engines and sixteen
  threads total; remaining fine drives use one sixteen-thread engine.
  All fifteen field solves have rc=0. At 1 GHz the 2.5-um reference gives
  L1/L2/M=1.786880/1.505557/1.061488 nH and k=0.647171; the 5-to-2.5-um
  change is about 3% in self-L and 0.01934 in k. Maximum magnitude S change
  is 0.04015. Observed self-L orders are negative; reference convergence at
  the qualification targets remains open.
  The quasi-static inductance reference is primary for L1/L2/M/k and winding R
  at its acquired frequencies (R width-filament spread up to 3.86% exceeds its
  3% target), while the open-source FDTD reference is evidence-only for S/Q/SRF/
  shunt G and must not supply L or k targets.
  No Richardson or other extrapolated limit is claimed for either reference.
  Three complete native coarse runs per PEC/Cu and vacuum/10-S/m variant
  exercise passivity, reciprocity and torch.equal S/current determinism.
  Full mesh work is 16*(12*N^2+F^2)=41604528832 bytes (38.75 GiB)
  before the unchanged 20% workspace reserve; its refusal is recorded.
  Full-mesh physical qualification and the <=1-min/frequency target remain open.
  Frozen future qualification targets: L 2%, Q peak 5%, |S| absolute 0.01,
  phase 1 degree, k absolute 0.01. Coupling uncertainty must be below its
  target before native qualification; finite-grid references are not exact.
  All thirteen new tests pass through the prescribed GPU runner and memory cap,
  rc=0, no skips/xfails; maximum GPU-test time/allocation are 96.291 s/0.488 GiB.
  The CPU guard census also passes all seven tests, rc=0.
  Frozen reference evidence is tests/mom/layered/data/fixture_fb_reference.json
  (173296 bytes), with input/binary/log/raw hashes and successful receipts.
  Drivers: scripts/dev/rfic/{reference,inductance,summarize}_fixture_fb.py.
  Evidence and unmet items:
  report.md of run fb in <evidence-root>/rfic-mom-20261106/.

- 2026-11-09: Evidence numerical locks (user decision D-P1), replacing source-hash pins.
  Rule: the committed layered evidence values (tests/mom/layered/data:
  fixture_r6_native_evidence.json, fixture_r6b_shunt_evidence.json,
  fixture_r4b_closed_comparison.json) are locked by value with the tiers of
  numerical_equivalence_standard.md. Source sha256 pins are not gates and are removed from
  the evidence; geometry digests (checked against a live compile) and data-artifact digests stay.
  Locked fields: R6 coarse/finest metrics and unknowns, the R6 pi effective shunt conductance,
  the R6b native_coarse/previous/fine records (observations excluded), and the R4b case
  unknowns and comparison samples (L, R, relative errors, pass flags).
  Tiers per field (committed value vs re-acquisition):
  - T0: float64 bits equal (discrete fields equal);
  - T1: max|new-ref| <= 1e-10*max|ref| per field, the fp64 run-level bound (network S/Y/Z and
    quantities derived from them); auto-accepted and listed with max_rel_diff in the merge
    declaration;
  - accepted T2: outside the bound and named, field by field, in an accept file with the user
    approval and its analysis;
  - T2: anything else, and every discrete change. A T2 blocks renewal.
  Derived relative-error fields (for example R4b L_relative) amplify the relative change of
  their input by 1/|value|; they are judged by the same per-field bound.
  Procedure (scripts/dev/rfic/evidence_lock.py):
  - acquire DIR on GPU 0 through the runner, on a clean witwin/ tree; DIR/acquisition.json
    records the commit;
  - judge DIR on CPU writes DIR/tier_table.{json,md} (with the merge-declaration text) and
    DIR/fields.json; exit 1 on an unaccepted T2;
  - renew DIR --accept-t2 FILE writes the moved values (T1, accepted T2) into the committed
    evidence and appends a renewal record to each file's numerical_lock (acquisition commit,
    tier table path and sha256, summary, renewed fields, accepted T2 entries).
  Producer artifacts outside the repository (R6 MPIE sweeps, R4b closed cases) are judged
  report-only; their pi-fit diagnostics are ill-conditioned nonlinear fits, not committed values.
  Fast tests: the CUDA evidence tests keep their fixture, physics-target and live-coarse checks
  without source hashes; test_layered_evidence_lock.py (host only) checks the lock records and
  that no source-hash pin returns. Heavy re-acquisition is a merge-time script run, once per
  merge that changes layered numerics or their producers. A renewal with T1 movers only needs no
  user approval; every T2 needs one, per field. The consumer-metadata bridge
  (tests/api/public/test_consumer_metadata.py) is a provenance gate and keeps its own re-receipt
  procedure.

- 2026-11-09: Horizontal admission floor (vertical tables). horizontal_z_error and
  horizontal_rho_error divide each row's error by max(row max|expected|, 1e-6 x the
  per-frequency family max), the vertical_error rule. Cause: on a stack PEC plane the
  tangential kernel cancels against its image to denormal noise (2.6e-321 beside a
  5.4e-2 family at 1 GHz, h = 254 um), so a PEC-endpoint row read as relative error 1
  and refused an accurate table (12 of 91 rows; 8..32 z nodes all refused). Table
  values are unchanged: nine previously admitted tables are bitwise identical with
  identical admission errors.

- 2026-11-09: C1 port system, C2 device-parameter extraction, A3 groundwork.
  Port V/I contract (unchanged sign chain, network.py): every port is an ideal
  lossless bridge with a delta-gap source on its midpoint cut; V = phi(+) - phi(-)
  is the same on every parallel path across the bridge; I is the total cut current
  leaving the positive terminal. LumpedPort.voltage_aperture is now applied: the
  bridge spans the aperture's transverse width (edge port), which is the FDTD meaning
  of the field (voltage over parallel paths, one current); FDTD is unchanged. The
  aperture must equal the terminal span along the path, contain the path in its
  width, have both bridge ends covered by metal over the full width (horizontal) and
  an open gap. Vertical apertures have zero extent along the strip normal. The
  reference is explicit through the negative terminal: a coplanar conductor, a stack
  PEC plane, or a conductor on the adjacent plane (sheet boundary attachment only).
  port_system.edge_port builds such ports from named PlanarPECs;
  metadata["ports"] records kind, orientation, conductors, width, length, cut and
  the V/I definitions. Closed-model aperture ports and rotated apertures refuse
  (guards registered). PortPair/mixed_mode: Vd = Vp - Vn, Id = (Ip - In)/2,
  Vc = (Vp + Vn)/2, Ic = Ip + In. extraction.py: terminals are ports or (p, n) pairs,
  Z_w = D^T Z_S D with the other ports shorted or open (no default); L = -Im Z/omega,
  R = Re Z, Q = -Im Z/Re Z, M, k, SRF (linear reactance zero), peak Q, pi/T models,
  T fit (closed form), nine-element pi fit (bounded projected Levenberg-Marquardt,
  implicit-function gradients). deembed.py: two-line gamma through identical
  reciprocal error boxes (branch continuity in frequency), open-short de-embedding
  and open/short standard Scenes from named structures.

- 2026-11-09: B1/B2 polygon layouts (general planar mesher, polygonal and multi-plane vias).
  Manhattan inputs keep the rectangle path (T0, unchanged code path); non-Manhattan PlanarPEC,
  z-axis Prism vias and vias passing a metal plane (or planar_mesher="triangle") compile to
  PolygonSheet (one per PlanarPEC, holes kept) and PolygonVia (footprint loop, exact plane z)
  records; prepare_layered_network dispatches them to planar.prepare_planar_network.
  Mesher (planar_mesh.py, host, deterministic): per plane, the noded union of material-group,
  bridge and cut linework restricted to the metal domain, plus via rims (not noded, must clear
  other linework), forms one constraint registry with plane masks. Conforming Delaunay
  refinement (Qhull): split missing or encroached segments (midpoint; concentric shells from
  acute input vertices), else insert circumcentres of triangles with min angle < 28 deg or
  longest edge > sqrt(2) s(centroid), s = min(h, h_edge + (g-1) d); circumcentres that encroach
  a segment split it instead; batch insertion keeps candidates > R/2 apart. Rim splits apply to
  every plane of the rim. Walls: rim segments (<= min(h_edge, 2 max dz)) times the graded z axis
  of the via (its planes and interior interfaces). Exact-coordinate welding; three-face landing
  rims switch the layout to the junction RWG basis. Quality gate: sheet min angle >=
  min(18 deg, half the smallest acute input wedge), aspect <= 6 on sheets and walls; acute input
  corners < 18 deg are refused. The mesh sha256 (vertices, faces, planes, owners) is recorded.
  Loss: per-face owner records (sheets then vias), two-face sheet impedance and single-face wall
  impedance as before; bridge-only faces are ideal. Ports: horizontal delta-gap bridges whose
  cut must clear metal; vertical ports and the closed model are refused on this path.
  Validation: tests/mom/layered/test_layered_planar_{mesh,assembly,network}.py; drivers and
  logs under <evidence-root>/rfic-mom-20261106/b1/.

- 2026-11-11: R5-1 compressed surface-admittance body blocks on matrix_free/hmatrix
  (body_compress.py, body_block_lu.PatternBodyLU). Action: each W_b is an H-matrix on the exterior
  RWG centres (median bisection, LEAF 32, admissibility min(diam) <= dist, dense near leaves); every
  admissible block is a batched full-pivot cross approximation stopped at an explicit residual
  |R|_F <= 1e-10 |W_b|_F, so the reported dropped mass is the exact |W_b - W~_b|_F. Applied through
  row-owner CSR products (Q_b, V^H, [N | U], Q_b^T), exact transposes for A^T; deterministic (ranks and
  factors torch.equal on repeat). Looped per-block GPU SVDs took minutes per F-B body and were replaced.
  Preconditioner: the sparse near LU (D2 pattern) with D B_b D added only on the pattern's own keys;
  the R4-1 dense Schur set and its admission guard are retired. Dense B_b exists transiently per body.
  Layouts without bodies: Y, currents, steps, LU nnz torch.equal vs mom/r4-blockpc (strip sheet/closed,
  F-B 80/20 sheet; mf and hm).
  Measured: strip 40/4 mf/hm vs dense 1.5e-9/1.0e-7/1.5e-7 and 2.4e-9/1.5e-7/1.1e-7 in 6/7/9 steps
  (= R4-1), LU 0.88M nnz (R4-1 3.77M), repeat torch.equal. F-B 80/20 @1 GHz: 8 steps mf and hm (R4-1
  8), mf 8.1e-9 / hm 1.2e-7 vs dense, LU 9.37M nnz (1.0 GB est.), W ranks <= 61, stored 0.37-0.69 of
  the dense entries, torch peak 5.67 GB, device 6.8 GiB (R4-1 7.4). F-B 40/10: 9 steps (R4-1 9), mf vs
  R4-1 mf Y 3.3e-10, L/M/R within 2.4e-8 of the dense backend; ranks <= 53, stored 0.27-0.44 of dense;
  torch peak 11.17 GB (R4-1 12.38, dense backend 12.2 at two frequencies), set by the DSA stage
  (structures 6.46 GB + one body's 4.7 GB DSA transient; 8.54 GB after it); device peak 14.0 GiB
  (R4-1 14.0), from structure build and DSA fragments, not from the preconditioner.
  Report: <evidence-root>/rfic-mom-20261106/r5-wcompress/report.md.

- 2026-11-11: R4-1 dense body-block preconditioner for surface_admittance on matrix_free/hmatrix
  (body_block_lu.py). The G2 fold of the dense body blocks into the sparse near LU is removed
  (34M LU nnz for 17M block entries on F-B 80/20). The preconditioner matrix M (D-scaled near or
  D2-widened pattern plus the scaled B_b) is split into a dense set D (every body basis column plus
  every rest column the pattern couples to one) and the sparse rest S: one dense complex128 LU of the
  Schur complement C = M_DD - M_DS M_SS^-1 M_SD, the cuDSS LU (deterministic mode) on S only, and
  y = M_SS^-1 r_S, z_D = C^-1 (r_D - M_DS y), z_S = y - M_SS^-1 M_SD z_D, i.e. M^-1 exactly.
  Per-body LUs were measured first and rejected: a forward block Gauss-Seidel sweep took 30 block
  steps at 10 GHz on the 40/4 strip (18 with three sweeps) vs 9 exact; F-B 80/20 bodies share 68
  basis columns and the off-body pattern touches 7267 of 7272 columns, so an exact interface step
  is dense in N anyway. Admission: 2 * 16 |D|^2 bytes against available_work_memory (process cap),
  else UnsupportedFeatureError (guard registered, negative test). Layouts without admittance
  bodies keep NearFieldLU: torch.equal Y and currents vs e42174e6e on strip sheet, strip closed and
  F-B 80/20 sheet, matrix_free and hmatrix. Limit: |D|^2 grows as N^2 when bodies span the layout;
  F-C-scale thick metal needs compressed W or a per-component split of D.
  Measured: strip 40/4 (|D| 1942 of 1946, dense LU 60 MB) mf/hm vs dense <= 1.5e-7, 6/7/9 steps
  (= G2), repeat torch.equal. F-B 80/20 at 1 GHz (|D| = N = 7272, S empty): dense LU 0.85 GB,
  mf 9.0e-9 / hm 1.2e-7 vs dense in 8 block steps, device peak 7.4 GiB (G2 fold: 15.6 GiB; dense
  backend 3.9 GiB). F-B 40/10 (N = |D| = 12640, dense LU 2.56 GB; G2 fold ran out of memory at
  15 GiB): mf and hm run at 14.0 GiB device peak in 9 block steps, transformer L/M/R within 2e-8
  of the dense backend (12.2 GB torch peak), hm vs mf |dY|/|Y| 1.9e-8.
  Report: <evidence-root>/rfic-mom-20261106/r4-blockpc/report.md.

- 2026-11-10: G2 surface_admittance on the matrix_free and hmatrix backends (admittance_near.py).
  Each body block W_b (exterior RWG ids_b) enters the shared near operator M, the near action and
  sparse-LU preconditioner of both backends: Q_b = Q[ids_b, cols_b] keeps the c_b basis columns
  touching the body; the basis block -i mu0 S Q_b^T (W_b i/(mu0 w)) Q_b S (mu0 = 4e-7 pi as the
  dense fold) is formed with two dense real x complex products per frequency and its c_b^2 keys
  are appended to the near pattern (the stable-sort/segment-sum merge after the MPIE entries).
  A symbolic sparse product of a dense block would expand ~nnz(Q row)^2 entries per W entry.
  A = M + Far is the dense system with the exact MPIE near/far split; Far is unchanged. DSA
  blocks are not symmetric (5% on the strip), so the operator adjoint applies M^T through a
  transpose permutation of the symmetric pattern, and the GMRES stopping bound uses |X| for
  |A^-T B| (first order in the asymmetry; agreement with dense is the gate). Body blocks are
  computed once per frequency. The action applies them as dense products on their columns;
  only the D2 widened LU pattern carries their keys (values added after the MPIE segment sums,
  the merged-segment order). Memory: O(sum_b c_b^2): 16 B per entry in the action, 8 B keys
  and the sparse-LU fill in the preconditioner.
  A1 interior kernels: far/near launches split over pair ranges (admittance.LAUNCH), one owner
  per pair, outputs independent of the split; single launches of 5 s tripped context-switch
  timeouts on time-sliced shared devices.
  Measured: strip 40/4 (low/window/DSA) |dY|/|Y| <= 1.5e-7 (matrix_free and hmatrix), 6-9
  block steps; F-B 80/20 thick (N = 7272, c_b = 2400/2572/2188) matrix_free 9.0e-9 vs dense at
  1 GHz in 8 steps, but 6.2 GB torch plus a 9.7 GB sparse-LU estimate vs 2.7 GB dense: the
  sparse LU of the dense body blocks (34M nnz for 17M entries) dominates. Follow-up: per-body
  dense LU block preconditioner, compressed W. F-B 40/10 matrix_free does not fit 15.5 GiB.
  Refusals retired: LayeredSurfaceConfig and prepare_layered_network backend guards.
  Report: <evidence-root>/rfic-mom-20261106/g2-fastthick/report.md.

- 2026-11-09: A1c two-regime conductor_model="surface_admittance". Per body, t = largest
  smallest box dimension, x = t/delta. x <= 0.3: uniform-current local conduction J = e/z on S,
  z = P/(sigma A) - i w dL'_int P from the owner-box cross-section normal to its longest dimension,
  A_low = <e, e/z>, W = B A_low^-1 B^T (= z <g, g> for one z): R = l/(sigma A) exact on prismatic
  bars; dL'_int = mu0/(2 pi)(ln GMD_perimeter - ln GMD_area) (perimeter closed form, area by a
  Duffy split with exact radial integral). x >= 0.6: A1 DSA block unchanged (torch.equal).
  Between: A = (1-s) A_low + s A_DSA, s = C1 smoothstep of log x; blending W instead overshoots R
  by ~1% (one current through both impedances). Per-body blocks computed before the dense chunk,
  admission counts them (extra_per_frequency). Gradients refused. Measured on the strip 20/2:
  DC R +5.4e-10 (10/1 below), R within 0.71% of the 25x13 filament reference 10 kHz-10 GHz,
  window monotone (adjacent dR <= 0.23%, dL <= 0.033%); regime mismatch at x = 0.3: R 0.52%,
  L 0.22%; x = 0.6: R 0.19%, L 0.11%. Unmet: L 1% (-1.06% low, -1.19..-1.25% DSA; shared with
  closed/PEC; reference converged (Richardson 9/17/25: 0.03%), exterior 20/2 -> 10/1: -0.07%);
  100 GHz (radiating loop, QS reference invalid). Report:
  <evidence-root>/rfic-mom-20261106/a1c/report.md.

- 2026-11-09: A1 (R4c) conductor_model="surface_admittance" (dense backend only).
  Differential surface admittance per lossy body V (closed dV = exposed S + PEC contact
  faces C). Interior identity (outward n, J = n x H, M = e x n): T_m j + K'_m eps + G eps/2 = 0,
  T_m = i w mu0 (V_m - P_m/k_m^2), K'_m = <f, curl S_m[n x g]>, e expanded in RT on S
  (S-interior RWGs + S-side seam halves), Gram coupling <f, e> = G eps. Divided differences
  F = (G_c - G_b)/(k_c^2 - k_b^2): j_b = T_b^-1(-K'_b - G/2)eps, J_D = T_c^-1[-dk2 K'hat eps - dT j_b];
  scalar operators only in tree-tree blocks via exact face charges. W = R_x^T G_S Y_S^-1 R_x per body,
  keyed by body owner, folded before the loop/tree congruence. Native CUDA face-pair engine:
  joint Duffy touching rules (radial panels for |k_c| h >> 1) on the bounded remainders,
  closed-form static inner integrals (scalar, moment, gradient) on graded (touching) and
  adaptive 4^L (near) observer rules, polar near remainders, product far rules; deterministic
  tile gathers. Contacts: material-group contact faces (closed.cu contact_mesh); horizontal
  ports become ideal PEC block bridges (terminal thickness) with cap contacts and a
  mid-section ring delta gap. Rejected: rotated-RT e (X exactly singular -> zero-resistance
  modes), RT e broken at sharp edges (worse), Leontovich local updates, 1-D coth/csch (oracle
  only), composite-union background (no same-surface divided difference), FEM interior.
  Measured: strip 1e8-1e10 Hz R/L within 1.3% (L offset -1.2% mesh-independent and shared with
  closed at 10 GHz); DC R +0.5%. Unmet: DC 1e-6; low-frequency L (|k_c| h << 1): internal
  inductance error -0.46 nH (20/2) / -0.31 nH (10/1) from a +-60 nH cancellation amplified by
  sigma P^-1; follow-up A1b = potential-based (A-phi, SLIM family) interior formulation.
  References: S. Sharma and P. Triverio (IEEE AWPL 2020; IEEE TMTT 2022); U. R. Patel and
  P. Triverio (2016-2017); M. Huynen, D. De Zutter, H. Rogier (3-D DSA). Report:
  <evidence-root>/rfic-mom-20261106/a1/report.md.

- 2026-11-08: Layered MoM input contract (0.7.0 correctness, no silent ignores).
  compile_layered_scene refuses, in one NotImplementedError that lists every item,
  each declared input the layered solve would not apply: Scene sources, monitors,
  thin_wires, networks, lumped_elements, circuits, material_regions,
  electrostatic_terminals, charge_densities and internal_absorbers; Scene.symmetry;
  any boundary face kind other than "none"; Scene.coordinates; ports that are not
  LumpedPort; and the LumpedPort fields termination, reference_plane,
  voltage_aperture, current_surface and wire_binding. LumpedPort.current_surface is
  optional (the layered port current is the delta-gap MoM current) and is required
  only by FDTD field ports. Trainable Layer/HalfSpace fields are refused at
  construction. LayeredNetworkRuntime.run recompiles the Scene and refuses a
  compiled-input signature (LayerStack, sheet rectangles and records, vias, ports,
  device) that differs from prepare().
  Thickness semantics (documented, not refused): conductor_model "sheet" does not
  model geometric thickness; for LossyMetalMedium physical_thickness enters only the
  surface impedance, and a Material.pec() sheet is infinitely thin whatever its
  declared physical_thickness. "closed" models the conductor volume. Result.metadata
  and Result.network.metadata record conductor_model and
  thickness_modelled_geometrically.

- 2026-11-09: D1 H-matrix/ACA far field (operator_backend="hmatrix") and D2 body-aware preconditioner.
  hmatrix keeps the S2a system, near LU and block GMRES and compresses the far face-pair
  operator (vector 3F x 3F, scalar F x F) once per frequency. One support octree per z class
  (each metal plane, the walls); unordered block pairs over all class pairs, so cross-plane
  blocks are compressed too. Admissibility is lateral only (images can approach the observer
  plane): max diam_xy <= eta gap_xy (eta = 1) and gap_xy^2 >= NEAR_RATIO^2 h^2 (1 + 1e-6), so
  every near pair of pair_near/wall_near lies in an exact leaf block (integer-checked).
  Entries come from the S2a list kernel (bitwise the dense tiles). Face-pivot cross
  approximation (rank <= 3 per vector sample, complete pivoting in the pivot block, random
  check face) with balanced Gram/eigh recompression at ACA_TOLERANCE = 1e-6 per block and
  channel: per entry at the float32 rounding floor of the dense far vector moments and inside
  the Rule 2 setup bound 64 eps32 max|ref|. Blocks are stored once and applied with their
  transpose (exactly complex symmetric); native low-rank/leaf action with fixed-order sums.
  The worst-case and quadrature compression bounds are metadata diagnostics; acceptance is
  measured |dY|/|Y| against dense; hmatrix_exact_check=True verifies each solve with one exact
  far action (refused above 1e-5). Measured |dY|/|Y| vs dense <= 1.7e-6 on F-A test/fine, F-B
  test/fine (1 kHz-100 GHz) and closed F-A fine; F-B 27k/37k/44k RWG agree with matrix_free to
  4e-8 with 2.2/3.1/3.8 GB compressed storage (O(N log N)-like) and an action 0.06-0.11% of the
  matrix-free far action. D2: on a closed body mesh the LU pattern adds same-body (face owner)
  pairs within WIDE_RATIO = 3 cells; the action is unchanged (Y within 7e-8); closed F-A fine
  steps 30/39/60/77 -> 16/22/30/36 at 1.26x LU fill. Hook commit 1f4c64f8 is a bitwise refactor
  of the matrix-free near values plus optional pair-list/preconditioner arguments.
  Drivers: scripts/dev/rfic/bench_hmatrix_d1.py.

- 2026-11-09: E1 frequency-dependent (dispersive) layer permittivity.
  Layer/HalfSpace.eps_r takes a constant (unchanged) or a causal model:
  WidebandDebye(eps_inf, delta_eps, f1, f2) (Djordjevic-Sarkar, exp(-i w t):
  eps = eps_inf + delta_eps ln((f2 - i f)/(f1 - i f))/ln(f2/f1), a uniform log-spread
  of passive Debye poles, Kramers-Kronig exact; from_loss_tangent(eps_r, tan_delta,
  frequency, f1=1e3, f2=1e12) matches Re eps and tan delta at one frequency) or
  PolePermittivity(eps_inf, poles) over the existing DebyePole/DrudePole/LorentzPole
  (from_material adapts an isotropic electric-dispersive Material, e.g. a
  fit_dispersion result; static conduction stays in Layer.sigma). Refused as
  non-passive/non-physical: delta_eps < 0, eps_inf < 1, not 0 < f1 < f2, non-finite,
  gain Lorentz poles, custom (spatially varying) poles, trainable tensors.
  Green table: a dispersive stack stores media as [F, regions] rows, eps(f) evaluated
  per table frequency at build; constant stacks keep the [regions] row (bitwise). Every
  CUDA permittivity read offsets the row by the frequency index of the table it reads
  (stride regions, or 0 for a 1-D row). _frequency_view slices the rows with the other
  frequency tensors, so PH1 dense chunks, the S2a matrix-free per-frequency views and
  adaptive probes never mix frequencies; admission stays per (frequency, pair,
  component) row. Host image pruning, the wave-number scale km and the uniform
  (exact-mode) test evaluate every frequency of a dispersive stack. Gradients through
  dispersion parameters are not supported (fixed inputs).
  Driver: scripts/dev/rfic/e1_dispersive_microstrip.py.

- 2026-11-09: S2a matrix-free backend (GMRES with a sparse near-field preconditioner).
  LayeredSurfaceConfig(operator_backend="matrix_free") solves the PH1 equilibrated
  loop/tree system without N^2 storage. Face pairs split by the assembly's own near
  predicate (pair_near / wall_near), which partitions every tile exactly. Near pairs
  are evaluated per frequency by the identical pair kernels over an explicit pair list
  (bitwise the dense tiles), gathered to V_near (with R/(-i mu0 w) folded) and P_near,
  and transformed by symbolic expand/stable-sort/segment-sum products to
  M = D[S Q^T V' Q S + C^T P C/(eps0 w0^2)]D, D = |diag M|^(-1/2) (Y and currents are
  invariant under D). M is the near action and a complex128 sparse-LU right
  preconditioner (GPU direct solver, deterministic mode; that mode admits one RHS per
  solve, and the default mode is not bitwise for several). Far pairs are re-integrated
  in every action by the same tile kernels in a far-only selection and contracted for
  all ports per tile visit. Block GMRES (CGS2, restart 48) stops when
  |X|_F|R|_F <= 1e-5 |B^T X|_F: Z is complex symmetric, so dY = X*^T R; a true residual
  confirms. The matrix-free loop/tree setup emits sparse cycle columns directly (no
  transient [N,N]); the dense setup and every dense result are unchanged (T0 checked).
  Each far action costs about one dense assembly; far-field compression (2b) is separate.
  Drivers: scripts/dev/rfic/bench_matrix_free_s2a.py.

- 2026-11-08: PH1 dense operator protocol and memory.
  Each frequency owns one complex128 equilibrated loop/tree system and one complex64
  workspace. The tile gather writes V at its loop/tree basis position; lossy Gram terms
  fold in as R/(-i mu0 w). The sparse congruence and the diagonal equilibration run
  in place, keeping the old summation orders; the tree-tree charge block reads packed
  P (bitwise symmetric through canonical face pairs) on the fly. The workspace holds
  packed P during assembly and the column-major complex64 LU afterwards (LU aliases A).
  Footprint 16N^2+8max(N^2,F(F+1)) (<=24N^2 for F<=N) plus O(F) tile/wall scratch;
  admission uses it. LoopTree persists only norms and CSC columns. Section 5's
  precision policy is unchanged (complex64 LU, complex128 refinement, complex128
  escalation only on failure, explicitly admitted). LayeredSurfaceConfig gains
  operator_backend="dense"; each frequency is a SurfaceOperatorProtocol operator
  driven by the shared Krylov interfaces. Pinned movers vs 74f2d6aa are T1 at
  rounding level (division replaces the vendor diagonal trsm; loss folding). The
  source-hash provenance pins await their own renewal procedure.
  Drivers: scripts/dev/rfic/bench_dense_ph1.py, evidence_lock.py (then evidence_compare_ph1.py).

- 2026-11-08: GR1 native adjoint network gradients for local loss parameters.
  Network extraction already returns X = Z^-1 V for all ports, and
  dY/dtheta = -X^T (dZ/dtheta) X, so no extra solve runs. Every GR1 parameter
  enters only R = sum_f Zs_f G_f: PlanarPEC/via LossyMetalMedium conductivity,
  Hammerstad/Huray roughness fields and sheet-model PlanarPEC thickness, for
  conductor_model "sheet" and "closed". A PyTorch-native map evaluates the
  per-group Zs from the Scene tensors (the prepare_sheet_loss and closed
  Leontovich formulas); the native loss_adjoint operator contracts the local
  Gram rows with L = X conj(gY) and X, one block per (frequency, group), with
  fixed-order face/row CSR walks and a fixed shared-memory tree: no dZ storage,
  no floating-point atomics, bitwise-repeatable gradients. Result.network keeps
  the forward values (S via one S->Y->S round trip) and gains the autograd graph.
  Reference impedances and port loads differentiate through NetworkData
  renormalize/terminate; a trainable LumpedPort.reference_impedance remains
  refused by the shared non-FDTD Simulation admission. Closed-model thickness
  (moves body faces), geometry, process-stack and termination gradients,
  adaptive sweeps with gradients and current gradients refuse explicitly.
  With no trainable input the forward path is unchanged. Gates: complex128
  gradcheck of the re-solving Zs -> Y map, dense contraction oracle (1e-12),
  central differences in the bound 2(|FD_2h-FD_h|/3 + 2 kappa beta max|Y|/h)
  for nine sheet/closed cases, three-run torch.equal, a SceneModule + Adam Q
  rise checked by an independent forward, and F-A all-Y gradients <= 2x forward.
  Report: <evidence-root>/rfic-mom-20261106/gr1/report.md.

- 2026-11-07: R4b opt-in closed conductor surfaces; physical qualification remains open.
  LayeredSurfaceConfig(conductor_model="closed") extrudes positive-thickness
  PlanarPEC objects symmetrically about their declared mid-planes. The default
  "sheet" and its two-face finite-thickness impedance are unchanged. Native CUDA
  builds the exposed rectangle/via union, including polygon-hole walls; each
  via joins the top face of its lower body to the bottom face of its upper body.
  Lossy exterior faces use single-face Leontovich impedance with the public
  complex roughness factor. PEC faces and top-face ideal bridges have zero loss.
  At a k-face edge, k-1 pair RWGs span the complete zero-sum integrated-current
  space. Native padded incidence, fixed-order potential/loss gathers and the
  loop/tree forest cover ordinary bends, three-face bridge rims and a tested
  four-face junction without floating-point atomics.
  Horizontal bridges attach to top faces while declared terminals retain their
  mid-plane coordinates. Zero thickness, closed vertical bridge ports, overlapping
  distinct-plane bodies, nonuniform via rims and unequal bridge top heights
  explicitly refuse; existing sheet/via geometry refusals remain. Five capability
  boundaries and two contract validations have negative controls in the guard manifest.
  Derived heights within eight fp64 ULPs of a dielectric interface canonicalize
  to it. Horizontal Green rows use the upper-region limit; tangential continuity
  is checked against an upward 10-pm displacement. Native affine centroids retain
  constant coordinates exactly, preventing a face from rounding outside a vertical
  Green segment. Metadata records the model, per-PlanarPEC union-owned face
  counts, shortened via gaps and via face counts; overlaps belong to the earliest
  declaration. New kernels are registered native PyTorch operators.
  Closed SIBC requires skin depth much smaller than thickness and width. It has
  no finite-conductivity DC resistance limit and is not a thin-lossy-metal model.
  F-A vacuum L(1 GHz)=3.1821 nH versus the resolved-copper bridge reference
  3.1503 nH is within 1.5%; L(0.1 GHz) and resistance miss. Longitudinal refinement
  from 10274 to 11138 unknowns barely changes these values; transverse convergence
  and original R6 physical qualification remain open.
  Original R6 tests, both strict xfails, frozen source/reference receipts and the
  consumer-metadata archive are unchanged. Six R6/R6b source-hash failures need
  re-receipts rather than rewritten historical evidence. Final native operator
  contracts and the 7/7 guard census are recorded separately.
  Drivers: scripts/dev/rfic/closed_fixture_r4b.py and closed_reference_r4b.py.
  Model, A-D tables, skin regimes, reference sensitivities, memory refusals,
  timing and unmet items:
  report.md of run r4b in <evidence-root>/rfic-mom-20261106/.

- 2026-11-07: R6b inductance and substrate-conductance arbitration.
  Preserve the original R6 targets and both strict xfails. The native finite-thickness
  impedance still acts on zero-volume sheets; a resolved conductor cross section is a
  different magnetic/electric model. The quasi-static inductance reference uses the
  compiled F-A metal union, finite ring, signal vias and copper conductivity.
  Explicit declared-terminal and native bridge-cut controls separate port parasitics.
  The 2.5-um direct volume grid gives L(1 GHz)=3.141291/3.150318 nH at
  declared terminals / bridge cuts; the last bridge-grid change is 0.50%.
  Direct matrices check the accelerated coarse solve to 0.031% in L; filament-height,
  filament-width and longitudinal segment controls are retained with the raw inputs.
  Vacuum native L(1 GHz)=3.308786 nH changes by only about 0.072% with the substrate;
  the 0.1-um sheet control reduces the disagreement with the volume-grid reference
  to about 0.4%. A first-order magnetic-thickness term is diagnostic only; it does
  not supply the missing closed surfaces or electric charge geometry.
  The 2.5-um PEC open-source FDTD reference fits in memory: L(1 GHz)=3.052896 nH
  and summed shunt G(3.4 GHz)=0.576693 mS, versus 2.982712 nH / 0.640093 mS at
  5 um. Retain finite-grid and temporal spreads rather than treating either mesh
  as a continuum answer.
  An isolated PEC plate/ring on the exact stack separates oxide-coupled silicon loss.
  Native G(3.4 GHz)=119.624 uS at 5586 unknowns; independent electrostatic rectangular-
  pulse collocation gives 118.217 uS at 9792 cells (1.19% difference within the
  observed mesh spread). Strict 300-ps plate open-source FDTD controls give
  139.330/129.889 uS at 5/2.5 um; their conditional first-order limit is
  120.448 uS. The simple area/spreading-resistance pi estimate is
  approximate and is not a qualification reference.
  Exact-stack scalar and all vector Green components cover source/observer heights
  300.5/302/305 um, 1/3.4 GHz and rho=0.1..500 um. The open-source layered Green
  function library's default cross-height integration errors reach 0.93% in row norm;
  the independent unextracted real-axis oracle agrees with native to 2.1e-11.
  Frozen data retain both the raw library values and the independently checked rows.
  Full layered validation: 182 passed, two unchanged strict xfails; four new controls
  pass, including three torch.equal complete runs of S and currents. Maximum full-suite
  test time is 81.023 s and recorded allocation is 3.475 GiB under the 6-GiB cap.
  No solver source, protected module, original R6 evidence or consumer-metadata archive
  changes. Physical R6 qualification remains open. Drivers and checks are under
  scripts/dev/rfic/ and tests/mom/layered/test_layered_fixture_r6b.py; complete evidence:
  report.md of run r6b in <evidence-root>/rfic-mom-20261106/.

- 2026-11-07: R6 substrate/shield reference study; physical qualification remains open.
  Extend the single F-A builder with none/pgs/solid shields at z=300.5 um,
  0.2-um doped poly at sigma=5e5 S/m (10 ohm/square), and a 220x220-um
  active footprint. Four orthogonal comb quadrants use 10-um fingers at
  transverse centres -80,-50,-20,20,50,80 um; diagonal slots avoid an
  under-coil closed loop. A perimeter outside the winding connects through
  lateral straps and a lower ring to the original top ground ring. Two
  staggered consecutive-plane via arrays traverse short intermediate ring
  landing sections: coincident tube rims would need unsupported three-face
  junctions; an extra closed intermediate ring is unnecessary for grounding.
  Coarse max_edge/edge_cell=80/20 um gives 4125/4456 PGS/solid unknowns;
  finer 20/5-um evidence uses 8935/10054 unknowns outside pytest.
  Independent open-source FDTD XML acquisition uses the same physical
  sheets/materials/vias, lossy silicon and oxide, lumped bridge-gap ports,
  graded feature-aligned meshes, absorbing boundaries and explicit runtime
  library paths. Preserve lateral/vertical resolution, boundary, source-gap and
  observation-window spreads; a matched PEC control measures finite-conductivity sensitivity.
  Compare full Y-derived L, Q and resonance with the original targets;
  never infer qualification from regression or determinism alone. The chosen
  10-ohm/square PGS decreases Q in both native and independent-reference data. Positive single-pi
  fits retain active component bounds and errors; fitted shunt loss includes
  shield heating and is not a unique silicon/metal loss partition. No solver
  source or protected shared module changes. Frozen reference and report:
  tests/mom/layered/data/fixture_r6_reference.json, source-bound finer-native
  evidence in tests/mom/layered/data/fixture_r6_native_evidence.json, and
  report.md of run r6 in <evidence-root>/rfic-mom-20261106/.
  Measured: peak-frequency 3.4% and SRF 4.0% pass; L(1 GHz) +6.1% and Q peak
  +28.8% miss (PEC control L +7.2%, effective shunt G -22.7%). Tests assert the
  passing targets, PGS direction agreement and the increase of the fitted Cox-coupled
  shunt conductance; the L/Q targets and the PGS delta-Q magnitude are strict xfails
  owned by R6b (independent quasi-static thick-conductor L/R reference, reference
  mesh convergence, isolated lossy-substrate shunt check).

- 2026-11-07: R7 process/GDS and adaptive network front-end decisions.
  ProcessStack.scene_from_gds returns a standard Scene with LayerStack background;
  LayerMap declares metal z, physical thickness and material, ViaMap names two
  consecutive planes. This keeps process metadata next to its importer without
  introducing another solver entry. GDS library.unit converts user coordinates
  to SI; library.precision sets Boolean/database precision. Flatten references
  and repetitions, canonicalize xy to that grid before exact Manhattan predicates
  (including quarter-turn transform roundoff), union each plane, undo keyhole cuts into PlanarPEC holes,
  retain one Box per via rectangle. Pin labels on a declared (layer,texttype)
  use name:+:metal and name:-:metal, including explicit negative ring/ground
  terminals; explicit LumpedPorts remain supported. Strict unknown mappings and
  malformed/unpaired terminals raise ValueError; non-Manhattan polygons,
  non-rectangular vias and skipped-plane vias raise NotImplementedError.
  Opt-in breakpoint_tolerance_m complete-span clusters share one x/y map over
  sheets, terminals and via rims once at scene compile (port bridge cuts derive
  from the clustered terminals), then round cluster means to that
  tolerance lattice (ties to even). z and dielectric interfaces remain exact;
  zero preserves the original path. Choose tolerance <= minimum feature/10;
  the jitter gate uses 5 nm for independent +/-2 nm vertex perturbations.
  Mesh quality reports minimum angle, longest-edge/altitude maximum and smallest
  edge in SI; the F-A gate bounds the aspect ratio by 20.
  LayeredSurfaceConfig.sweep=AdaptiveSweep(f_min,f_max,tolerance,max_points)
  owns frequencies through Simulation.mom. Nine linear seeds, shared-pole
  NetworkData.fit_rational S fits and nested-model disagreement select greedy
  interval bisections; an independent high-curvature midpoint solve validates
  convergence. Solves used for validation count against max_points. Result.network
  retains solved samples and currents; Result.surface_currents.sweep holds the
  fitted model and 2000-point dense NetworkData for existing Touchstone export.
  Passivity uses the shared pole-aware check plus dense sampling; violating
  models are rejected without changing residues/constants. Tolerance is a model
  change/error estimator, not an all-frequency mathematical certificate.

- 2026-11-07: F-A fixture frozen; physical gates A-D pass, M2 performance qualification remains open.
  Builder: tests/support/rfic_fixtures.py, exclusively Scene -> Simulation.mom -> Result.
  Square spiral: 3.5 turns, outer/inner side 200/105 um, width/spacing 10/5 um,
  sigma=5.8e7 S/m, top/lower PlanarPEC physical_thickness=2/1 um.
  Silicon: 300 um, eps_r=11.9, sigma=10 S/m, continued into a matching
  half-space to avoid a PEC backside image. Oxide: 6 um, eps_r=4.1;
  lower/top metal are 2/5 um above silicon. Two 4x4x3 um Box vias connect
  z=302/305 um. The lower underpass runs from (-50,-50) to (125,-50) um;
  top terminal leads end at (125,95) and (125,80) um.
  Closed 10-um ring: inner/outer bounds +/-145/155 um, 45-um turn clearance.
  Two x gap ports bridge x=130..145 um at y=95/80 um; the no-ring
  one-port variant bridges y=85..90 um at x=125 um. Bridges are ideal.
  Square-capped strips include corner/via pads without redundant internal
  centreline breakpoints; the physical metal union is exactly unchanged.
  Test/full max_edge, edge_cell=(20,5)/(10,4.9) um, growth_ratio=1.5:
  2953/6041 unknowns, 4532 full-mesh triangular faces.
  exp(-i omega t): Zs=-1/Y21, L=-Im(Zs)/omega, R=Re(Zs),
  Zin=1/Y11 and Q=-Im(Zin)/Re(Zin).
  A: full one-port PEC L(1/10 MHz)=3.386420332/3.386420540 nH.
  Independent 16-sub-strip Greenhouse/Ruehli reference=3.432504606 nH,
  error=1.342584% (<2%); 16->32 sub-strip change=0.1033%.
  The plain single-strip sum=3.455732696 nH (2.0057% difference),
  reported separately. Spiral-only current-sheet/modified-Wheeler values
  are 2.925734235/2.958852242 nH; lead/underpass contributions are excluded there.
  B: full R(1 MHz)=2.229071991 ohm vs DC path=2.282327586 ohm,
  error=2.333390% (<3%); solid-via DC resistance is only 0.2833% of the path.
  C: 25 logarithmic frequencies, 0.1..20 GHz, through R5.
  Relative Y reciprocity=1.71810e-12, max singular S=0.999970434191.
  Three complete test-mesh runs are torch.equal in S and every current.
  Full L(0.1 GHz)=3.479225499 nH vs vacuum=3.479216211 nH,
  difference=0.000267% (<1%); silicon skin depth is 15.915 mm.
  Mesh changes normalized to full: L=0.090054%, Q peak=3.644125%,
  sampled peak frequency=0%; maximum series-L change over all points=0.5655%.
  Full Q peak=16.310843964 at 3.419951893 GHz. Interpolated Zin self resonance
  is 14.564916863 GHz, bracketed by 12.861081668..16.038130607 GHz.
  D: test-mesh Q peaks for silicon sigma=10/0.1/0 S/m are
  16.905231499/32.035527583/34.282683983. Doubling top thickness 2->4 um
  lowers series R(1 GHz) from 2.645814540 to 1.581397381 ohm.
  E: GPU-1 seconds/frequency: V/P=4.655015078, transform=0.072554569,
  LU/refinement=0.262406724, recovery=0.015706683; total=5.005683054
  (worst chunk=5.014054). The strict <=5 s/frequency sweep target is unmet.
  Three direct 1-GHz runs take 4.992952024/4.999914660/4.992513124 s,
  all <=5 s and torch.equal. One-time horizontal/vertical-span table costs
  are 2.031438477/7.786467773 s; full peak allocation is 19.760304 GiB.
  P1 logical kernel models and profiler traces are retained; several kernels
  remain below the binding R0 50% dominant-resource target.
  All 99 existing tests remain unchanged and pass; all 11 new tests pass
  (max 69.034170 s, 2.328682 GiB), runner rc=0. No solver implementation
  or public API changed. Frozen mesh evidence:
  tests/mom/layered/data/fixture_fa_convergence.json. Report, L/R/Q tables,
  timings, references and test artifacts:
  report.md of run fa in <evidence-root>/rfic-mom-20261106/.

- 2026-11-07: R3b adds consecutive-plane Box vias and boundary-attached vertical delta-gap ports.
  Via footprints are removed from both sheets; exact-coordinate welding makes four-wall tubes
  with ordinary two-face bent RWGs. A deterministic CUDA integer BFS orients the faces.
  Three-face edges, interior strip attachments, intersecting/touching vias, skipped metal planes
  and off-plane/unsupported-material declarations raise explicitly. The public compiler snaps each
  Box bound to a declared sheet coordinate within 1e-6*(|position|+size) on that axis (a float32
  micrometre Box misses its planes by ~1e-11 m); the internal path checks exact identity.
  Walls use the bulk single-face Leontovich impedance (1-i)/(sigma*delta), rather than the
  two-face finite-thickness sheet impedance. Port bridges remain ideal/lossless.
  Wall coupling contracts H*(fx*gx+fy*gy)+Azz*fz*gz+A_zrho*fz*(dhat.g_h)
  +A_rhoz*(f_h.dhat)*gz, with observer-minus-source radial displacement.
  The one scalar D is used once; the R3a formulation-C correction is already in the vector terms.
  All near static rays use reflected/translated closed-form triangle integrals.
  For a radial cross ray set d=beta*(z-z_image), L=log(R+d), H_rho=delta_rho/[R(R+d)].
  The identity (r'_h-v_h).H_rho=(r_h-v_h).H_rho-1+d/R reduces its inner
  integral to the triangle area, the existing scalar/first moments and boundary integrals
  of L and 1/(R+d). On vertical faces, int 1/[R(R+d)] is
  beta*sum_edges m_z*int_edge 1/(R+d); on horizontal faces only edge logarithms remain.
  With R=sqrt(s^2+A), d=a*s+b and t=(s+R)/sqrt(A), both edge primitives are rational
  integrals with Q=(1+a)t^2+2*b*t/sqrt(A)+(1-a); definite log1p/atan2 formulas
  and exact vertical-edge/zero-radius limits avoid cancellation. Reciprocity evaluates
  the rhoz inner term in the exchanged order with coefficient -rhoz. Direct and PEC rays
  retain the exact outgoing-wave remainder. Canonical face order and the existing fixed gathers
  emit separate V/P blocks, preserving the original planar arithmetic and the R5 sign chain.
  Vertical tables additionally retain H at intermediate heights, with the same independent
  rho/z admission gates. For PEC endpoints, interpolate the remainder with an added endpoint
  constraint on the TOTAL kernel: cropped image lists need not vanish separately.
  If b(z)=product_j((z-z_j)/(z_g-z_j)), replace each Lagrange weight by
  l_j(z)-b(z)*l_j(z_g), and add the corresponding negative analytic-image endpoint value;
  tensor-product constraints include the corner inclusion term. This preserves each interior
  sample, exact PEC tangential/scalar zeros and arbitrarily small one-sided distances.
  Grounded bridges use 12 interior nodes; other declared spans retain eight. The CUDA assembly
  caches ordinary/constrained z weights once per face/rule point. P1 distance classes, fp64
  predicates, symmetric rules, far accumulation classes and deterministic tile/gather order remain.
  At a PEC ground only boundary half RWGs are admitted. Their image cancels the omitted
  boundary line charge, so physical plus image charge is neutral. An uncharged reservoir
  vertex completes the integer forest/cycle graph; it is never a Galerkin face. Port excitation
  and current use the same signed edge-length vector: current leaves the positive terminal,
  Y remains reciprocal and Im(Z_loop)=-omega*L. The strip and square-via equivalent radii
  and rectangular-loop partial-inductance derivation are in tests/support/layered_via_references.py.


- 2026-11-06: R5b bounds dense work by available-memory frequency chunks. Admission checks one
  frequency, independent of sweep length, with a 20% vendor-workspace reserve and the process allocator cap.
  D5 is restored: complex64 LU with complex128 residual/refinement; only nonconverged frequencies
  escalate to complex128 LU. Fixed device iterations freeze each frequency at its first passing step;
  one convergence/metadata transfer follows each chunk/precision attempt. The same per-frequency vendor
  LU entry points are used for every chunk size. Factor precision and refinement steps remain per-frequency
  metadata. This supersedes the R5 all-frequency complex128 solve decision below.


- 2026-11-06：R5 低频稳定化，默认网络路径统一使用分离势的 loop/tree 系统。
  - 取单位积分的 P0_t=1/area_t，D 为带符号边长（积分散度），P 为面对核的平均值；
    与密度散度及未归一化 P 的形式完全等价。原 assemble_layered 的求和顺序保留（T0）。
  - GPU 按边编号升序做确定性 union-find，森林用固定面/局部槽顺序 BFS 定向；
    每个非树边和唯一树路径生成一个基本环。非树边上的单位主元和树边主元证明完备性，
    包含孔洞与不连通分量。cuSOLVER QR 正交化后，前 N-(F-C) 列仍是无散环空间。
  - exp(-iωt)：S=[Q_L,(ω/ω0)Q_T]，ω0=c0/max_edge，
    A=SᵀZS/ω 的 LL=-iμ0 V_LL，LT=-iμ0(ω/ω0)V_LT，
    TT=i D_TᵀP D_T/(ε0ω0²)-iμ0(ω/ω0)²V_TT；标量势从不进入环块。
    导体损耗为 SᵀR S/ω，之后做对角合同平衡，端口 RHS 和电流恢复使用相同变换。
    Delta-gap 的 b_e/l_e 精确为 0 或 ±1；先在基本环/树上做整数环流求和，再用
    QR 的 Rᵀ 三角求解变换到 Q 空间，保留梯度型端口的精确零环激励，避免 1/ω 放大舍入噪声。
  - 修改第 5 节：默认 runtime 用单次批量 complex128 LU（无降精度/升级分支）；
    后向误差仍受 64 eps64 约束。原 dense solve 仅供现有独立参考测试使用。
  - 接受 f≥1 Hz，f=0 拒绝并明确 DC 为正频率极限：
    PEC 环 Z→-iωL，单位电压驱动的 I∼1/ω（ωI 有限；激励∼ω 时 I 有限）；
    电容 Z→+i/(ωC)，I∼ω；有限电导片趋于 DC 电阻矩阵。
    不把单位电压驱动的无损闭合 PEC 环解释为 f=0 的有限电流解。


- 2026-11-06：初版冻结。
- 2026-11-06：按用户要求，开源 MoM（X1）和开源 Green 函数库（X2）成为首选外部参考。
- 2026-11-06：R1b 之后，X2（开源 Green 函数库）接入完成。
  - 该库用 exp(+jωt)，取共轭后与本实现的归一化一致。
  - 它的默认积分精度约 3e-4，所以 X2 门限设为 max|G| 的 1e-3，只检查约定、归一化和分层。
  - 末位数字由 O2 把关：本实现和 O2 在有损四层结构上一致到 2.5e-12。
- 2026-11-06：R2b 修订第 5 节的收敛判据。
  - 原判据 ‖r‖/‖b‖ ≤ 1e-10 对低频 EFIE 不可达。cond(Z) 约为 1/(kh)²，1 mm 环、1 GHz 时约 1e8；即使用后向稳定的 complex128 LU，‖r‖/‖b‖ 也只能到 cond·eps 量级。
  - 改用逐列 normwise 后向误差 ‖r‖/(‖Z‖_F‖x‖+‖b‖) ≤ 64·eps64。
  - complex64 LU 加精化不收敛时，升级到 complex128 LU；所用精度和精化步数写入元数据。
  - 低频稳定化（loop-star 或 augmented EFIE）仍按计划放在后续阶段。
- 2026-11-06：R2b 端口细节。
  - 桥接方向取 LumpedPort 的 voltage_path（只允许 x 或 y）。
  - 切线上的边按边中点严格落在桥宽内来选取，所有切线边都必须施加激励：漏掉一条就会把间隙短路。
  - 符号链写在 network.py 的模块说明里。在 exp(-iωt) 下，电感满足 Im Z = -ωL < 0。
- 2026-11-10: G3 thin-strip L offset localized to the reference, no solver change. The A1c
  centreline FastHenry loop (one segment per run piece, corners joined at centreline nodes)
  over-estimates the 10x2 um loop by +1.19% (PEC, equal filaments) against a true-geometry lattice
  FastHenry model (cell-centred PEEC of the exact ring, graded to every edge, cut-plane delta gap;
  scripts/dev/rfic/strip_reference_fasthenry_g3.py): 4.8 pH localized at the corners, found by
  loop-length scaling (200/400/800/1600 um). Per-length L of the MoM matches an independent 2D PEC
  BEM (scripts/dev/rfic/strip_pul_bem_g3.py) to +0.25% (mesh, converging from above). Thick 20-um
  control: the centreline model is +1.84% high too; the earlier 0.05% "agreement" was that bias
  cancelling the closed model's zero-thickness top-face bridge. The strict expected failure is
  replaced by test_strip_inductance_true_corner_reference (1 MHz..10 GHz, 1%).
- 2026-11-11 R4-3 (thick-terminal ports): a horizontal `voltage_aperture` edge port on
  conductor_model="closed" now builds an ideal PEC block over the aperture width and the full
  terminal thickness (network._thick_terminal_blocks), welded to both terminal end faces by the closed union mesher;
  the delta gap is the block mid cross-section ring (the surface_admittance block port, shared
  _cut_vectors branch). This lifts the closed x edge-port refusal (guard 715430df retired). On the
  G3 strip loop at 1 GHz (PEC, 20/2 um mesh) the edge port gives 0.264885 nH for t = 20 um
  (-0.32% vs the lattice 0.265744; the top-face delta-gap bridge 0.268676, +1.10%) and 0.382023 nH
  for t = 2 um (-0.46% vs 0.383802). Closed delta-gap ports keep the top-face bridge bitwise; the
  edge port is their de-embedded form. Vertical ports on closed bodies stay refused (4991fdd0).
  surface_admittance already used the block; its 20-um edge port reproduces R = l/(sigma A) at DC.
- 2026-11-11: R5-2 A1 interior far kernel, one warp per far face pair (no output change). Lanes
  evaluate 32 consecutive (i,j,k,l) nodes of the product rule and store the 29 real factors of
  every accumulate() term in shared memory; each lane owns up to three of the 94 real output slots
  and adds the batch's terms in node order. Every term and every addition is the IEEE operation of
  the former one-thread-per-pair kernel (module built with --fmad=false; c10::complex scales
  componentwise), so all seven interior operators are torch.equal to it. The former kernel stays as
  the test oracle far_serial_kernel, reachable only through the test-only op
  admittance_operators_serial_far (tests/support/layered_admittance_oracle.py). Far launches hold
  1<<15 pairs (longest F-B 80/20 launch 0.71 s); far kernel 2.7x faster than the serial one there.
- 2026-11-11: R6-2 A1 interior kernels, divided-difference Taylor coefficients formed once per
  launch (no output change). For |kc| R < 1/2 the series of F = (g_c - g_b)/dk2 formed the
  r-independent factors c_n = i^n (kc^n - kb^n)/(n! 4 pi dk2) and c_n (n-1), n = 1..26, at every
  node (26 complex divisions). taylor_kernel now forms them once per admittance_operators call with
  the identical expressions; the per-node series keeps its r-dependent operations and order
  (--fmad=false), so all seven operators are torch.equal to the base build (strip 1e6/2e8/1e10,
  F-B 80/20 bodies). The Taylor branch dominated the near kernel at low frequency (strip 1e6:
  near time 1.8x that at 1e10 before, about 1.0x after; indicative, shared GPU).
- 2026-11-11: R6-2 thick-metal memory on matrix_free/hmatrix (no output change). (1) The near basis
  products V'Q, Q^T V'Q, C^T P, C^T P C and their merge are no longer stored per structure (the
  SparseProduct plans were 1.6 GB of index arrays on F-B 80/20): build_near_structure keeps only
  the merged pattern, and near_values replays the symbolic products per frequency in X row chunks
  of <= 2^23 expansion entries (matrix_free.sparse_product / merge_blocks). A row chunk's stable
  key sort is the global one restricted to its rows, and segment_sum adds each segment in order,
  so the values are independent of the chunk size and bitwise those of the stored plans. (2) The
  D2-wide preconditioner structure shares the action structure's Green operands, rules and
  assembly cache (the cache was built twice, 0.4 GB on F-B 80/20). (3) AdmittanceBody keeps R_x,
  the body cycle transform Q and the face charges as sparse COO of the exact dense entries and
  densifies them per use; the per-body Grams are formed per call after the interior operators
  (no longer cached for every body). SurfaceAdmittance.body_block(i, f) forms one body's W, and
  body_compress.body_operators streams bodies: W_b is folded, compressed and released before the
  next body's interior operators exist. body_impedance consumes its interior operators as it goes
  (congruence of a popped operator, in-place scalings and equilibration) with the same IEEE
  operations and operand order. Operator state (near values, D, body H-factors, M^-1 drive) is
  torch.equal to master e64438f0e on F-B 80/20 and 40/10 at 1 GHz. Measured (runner, one
  frequency, operator setup + one preconditioner apply): F-B 80/20 torch peak 5.67 -> 2.61 GB,
  device 6798 -> 3948 MiB; F-B 40/10 torch 11.17 -> 5.17 GB, device 14354 -> 7034 MiB (5892 MiB
  with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True); structures 6.22 -> 1.13 GB allocated;
  one body's DSA transient 4.7 -> 3.5 GB (the interior-operator call itself holds the seven
  operators plus its 1 GiB pair tile, whose size fixes the gather summation order). F-C class
  fixture: scripts/dev/rfic/fixture_fc.py (rows x cols array of F-B transformers; 40/10 is 12640
  unknowns per device, 4x4 ~ 2e5) and fixture_fc_series.py (one process per case).
- 2026-11-11 R6-1 (A2, default thick-metal model): LayeredSurfaceConfig.conductor_model defaults to
  "auto" (runtime.select_conductor_model), resolved once per Simulation by a physics-only rule:
  surface_admittance when max t/delta over lossy PlanarPEC at f_max (AdaptiveSweep.f_max for a sweep)
  is >= 0.3 (the lower edge of the surface_admittance regime window), sheet otherwise. No fallback:
  a surface_admittance selection on a layout with a polygon layout, zero-thickness PlanarPEC,
  roughness, a vertical port or trainable inputs raises (guard 5a2c81e7) with the feature and
  t/delta; conductor_model="sheet" accepts the sheet approximation explicitly. Metadata records
  conductor_model (selected) and conductor_model_selection. fixture_fa/fixture_fb pin "sheet" (the
  locked R6/R6b/F-B evidence is the sheet model; polygon thick-copper test configs pin it too).
  T2 behaviour change approved by the user 2026-11-11 (relayed by the RFIC lead; items: auto default,
  raise instead of sheet on refused features, "auto" payload, sheet pins, new surface_admittance evidence).
  R6 no-shield with surface_admittance (20/5 um, tests/mom/layered/data/
  fixture_r6_surface_admittance_evidence.json) vs the 5-um reference: L(1 GHz) +0.73%, Q peak -2.2%,
  peak frequency -6.0% (targets 3/10/10%). The sheet gaps (L +6.1%, Q +28.8%) are attributed by exact
  pi-branch swaps (scripts/dev/rfic/attribution_fixture_r6.py): series R at the peak 3.09 vs 3.77 ohm
  carries 2.7-2.9 of the 3.65 Q gap; the sheet strict xfails stay (locked sheet evidence).
- 2026-11-11: R6-4 vertical closed ports and TRL de-embedding (plan A3). A vertical port between
  closed (or surface_admittance) bodies on consecutive metal planes is an ideal PEC block from the
  lower top face to the upper bottom face, spanning the bridge width along the shared edge and the
  thinner terminal thickness into the metal; the mesher gets it split at its mid-height cut, whose
  ring is the delta gap. Vertical ports to a stack PEC ground stay refused (guard 5cec8f9a; the
  junction basis has no grounded half RWGs); guard 4991fdd0 is retired. Two-strip PEC line in eps_r = 4
  oxide: two-line eps_eff - eps_r <= 4e-4 relative; surface_admittance PEC == closed bitwise. deembed.trl: TRL in ABCD
  form, reciprocal boxes (det = 1), reflect root by kind, impedance level from a D = 1 lumped port
  model at both port planes plus the feed lengths (port_model_residual = left/right mismatch) or a
  given line impedance; deembed() works per port (no z21 division); trl_standards() builds thru,
  open reflect and line Scenes of a rectangular PlanarPEC two-port. Copper microstrip, 1 mm series
  strip, 1/2/4 mm feeds, 2-5 GHz: L spread 0.15%, Q spread 0.22% (1-2 mm feeds), 1.4% at 4 mm / 5 GHz (open).
- 2026-11-11 R7-2 (A4 certification, internal independent references only; no solver change): new fixtures
  F-D (MIM: equal plates on a 1 um eps-7 nitride film, vertical delta-gap port), F-E (PEC-ground microstrip
  h = w = 254 um eps 9.8; finite-ground CPW 10/10/40 um on a silicon half-space; edge ports) and F-F (two-turn
  octagonal differential inductor, apothems 70/58 um, w 8, s 4, underpass crossover, PEC, polygon mesher).
  De-embedding uses master's API only: two-line gamma (deembed.line_parameters) for eps_eff, length differencing for
  C' (MIM plates 60/120 um, open-ended lines at 10 MHz); Z0 and eps_eff are composed from separately extrapolated
  L' = eps/(c^2 C') and C'. References: 2D quasi-static spectral-domain Galerkin (quasistatic_sda_r7.py; exact CPW
  conformal map to 6e-12) and the FastHenry sub-strip lattice of the octagon (octagon_reference_r7.py) at n = 16,
  zero-thickness limit in h, calibrated on the straight trace pair against the exact 2D value (the lattice
  converges faster than first order beyond n = 16, so an n = 8/12/16 extrapolation overshoots by 0.6%):
  0.88718 nH +-0.18%. Results: F-D and CPW pass their derived tolerances; microstrip L' converges first order to
  +0.38% (C' to +0.02%): the air-filled line has the same L' error and exact C', and halving the longitudinal cell
  moves L' by +0.003%, so the offset is in the vector-potential path over the PEC ground (image / image-extracted
  table), not the dielectric or the mesh; F-F edge-only sequence non-asymptotic, +0.66% at 1 um. Evidence locked
  (D-P1 parts fd/fe/ff in evidence_lock.py; F-D on RTX A6000, F-E/F-F on L40S with their A6000 re-acquisition a
  pre-merge step).
- 2026-11-12: round-6 group merge (mom/round6), A4 re-lock. All three A4 files re-acquired with --repeat on
  RTX A6000 at the merged code head b6b88dd2b and re-assembled from those records (one device type for all locks):
  renewals T0 156 (F-D, records identical to R7-2's A6000 lock), T0 132 (F-E), T0 45 (F-F). F-F now uses the
  proportional sequence (set r2, maximum edge 2x the edge cell, 2/1.414/1 um; the first two meshes coincide with
  the edge-only ones): +0.644% at 1 um against 0.88718 nH +-0.18%, inside the derived tolerance 0.655% (no
  extrapolation: steps -0.06%/-0.47%) and the 1% target, so its strict xfail is gone; the one-off 0.707/1.414 um
  check (cloud RTX A6000, 22312 unknowns, does not fit a 15 GiB lane) gives +0.557%. The PEC-ground microstrip
  L'/eps_eff offset stays the only strict xfail (unchanged values class: +0.38%/+0.41%).
- 2026-11-12: R7-1 B2 arbitrary-outline vias. witwin.core ExtrudedPolygon (core feat/extruded-polygon,
  additive, Prism frame, exact scale-free SDF) is a third via primitive: via_record maps a z-axis
  ExtrudedPolygon (rotation about z only) to an exact counter-clockwise PolygonVia outline in fp64;
  compile_layered_scene and _needs_planar route it to the planar mesher. ProcessStack imports
  irregular GDS via outlines as ExtrudedPolygon on the database grid (rectangles stay Box, regular
  outlines stay Prism, bitwise); guard faae7b7a retired, 01bd8946/219df972 re-registered as
  f809e749/9f961005. A regular-octagon ExtrudedPolygon via loop is bitwise the Prism loop. Via-array
  reference: tests/support/via_array_reference.py, a 2D magnetostatic BEM (exact log panel
  integrals, graded panels; all vias of a net share A_z because PEC planes close every same-net loop)
  with the Leontovich wall condition A_z - Zs K/(i omega) = c_net (the model of the MoM via walls),
  Z' = -i omega (c_+ - c_-); exact against the coaxial (with impedance) and two-wire (PEC limit) closed
  forms to O(N^-2). Gate: a 2x2 via array inside an 8-via fence, de-embedded by the height difference
  60 -> 120 um: dR -1.16%, dL +1.54% (edge 4 um), -1.16% / +1.39% (edge 2 um, 21k/32k unknowns), KCL
  <= 3e-12; 40 -> 80 um gives -2.45% / +3.5% (end coupling through the pads ~ 1/h; the per-via split
  departs from the BEM's by about 2.7 um / h). The PEC-limit K overstates R' by 29% here (197.4 vs
  152.9 ohm/m): at Rs/(omega L') ~ 0.1 the wall resistance relaxes proximity crowding. Evidence: scripts/dev/rfic/via_array_reference_evidence_r7.py
  (closed-form N^-2 tables, both reference values, Zs -> 0 limit).
