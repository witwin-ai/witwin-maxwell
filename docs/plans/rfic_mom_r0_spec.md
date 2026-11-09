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
  <evidence-root>/rfic-mom-20261106/codex/r6b/report.md.

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
  <evidence-root>/rfic-mom-20261106/codex/r6/report.md.
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
  <evidence-root>/rfic-mom-20261106/codex/fa/report.md.

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
