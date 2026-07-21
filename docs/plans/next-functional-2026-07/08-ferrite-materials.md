# 磁化铁氧体与旋磁材料开发计划

> 状态：proposed  
> 日期：2026-07-14  
> 目标证据：E3 production  
> 类型：独立非互易材料与 FDTD 运行时能力  
> 前置依赖：现有磁场 Yee 更新、磁色散框架、对角 `mu_tensor`、CUDA adjoint；RF 端口只用于后期器件验收  
> Owner modules：`media.py`、`compiler/materials`、`fdtd/`、adjoint bridge、`result.py`  
> 最近架构决策：2026-07-14，使用 Polder/线性 LLG 的局部隐式磁化状态更新，不直接解除现有 full `mu_tensor` guard  
> 公共架构约束：`Scene + Simulation + Result`，PyTorch-native，GPU-first

## 1. 背景与当前能力

Maxwell 当前支持标量/对角 `mu_tensor`、磁 Debye/Drude/Lorentz poles 和标量静态磁导电损耗 `sigma_m`。这些模型仍是互易的：它们不能表达受 DC 磁偏置的铁氧体，也不能产生 Polder 张量的反对称耦合、圆双折射、Faraday rotation 或环行器的方向选择性。

现有 full off-diagonal CUDA correction 仅针对对称正定电介质；`Material` 也明确拒绝 off-diagonal `mu_tensor`。因此本计划不是给 `mu_tensor` 解除校验，而是引入有因果时域动力学的 gyromagnetic constitutive law。直接在每个频点填入复 Polder 张量不能用于宽带 FDTD，也无法保证稳定、无源或可微反向。

## 2. 目标与非目标

### 2.1 目标

1. 以 SI 单位的一等 `GyromagneticFerrite` 表达饱和磁化、偏置场、旋磁比、阻尼和高频背景参数。
2. 通过线性化 Landau-Lifshitz-Gilbert（LLG）/等价磁化 ADE 在时域产生正确 Polder permeability。
3. 支持任意均匀偏置方向；后续支持空间变化但静态的 bias tensor field。
4. 使用局部稳定的隐式中点/梯形更新和原生 CUDA kernel，阻尼非负时保证被动性。
5. 支持与电介质色散、电导率、非均匀网格、PML、端口和监视器组合。
6. 支持材料参数 autograd、精确 reverse recurrence、checkpoint 和 multi-GPU。

### 2.2 非目标

- 不求解产生偏置场的磁静态线圈/永磁体；用户输入的是已知静态 `H0` 或 `B0` 分布。
- 不在本计划中实现大信号磁畴翻转、磁滞、饱和非线性或温度依赖。
- 不把任意复 `3x3 mu(f)` 作为时域材料接受；必须具有可验证的因果、稳定状态空间。
- 不在第一版支持空间变化 bias、反铁磁体、多共振磁晶各向异性或 spin-wave exchange。
- 不增加其他全波后端 fallback；本计划只设计和验收 FDTD 时域旋磁更新。

## 3. 用户功能描述

用户以数据手册或实验可获得的物理量建立材料：

```python
import witwin.maxwell as mw

ferrite = mw.GyromagneticFerrite(
    eps_r=14.5,
    saturation_magnetization=1.40e5,  # A/m
    bias_field=(0.0, 0.0, 1.75e5),   # A/m
    gilbert_damping=2.0e-3,
    gyromagnetic_ratio=1.760859e11,  # rad / (s*T)
    sigma_e=0.0,
    name="biased_yig",
)

scene.add_structure(mw.Structure(geometry=ring, material=ferrite))
result = mw.Simulation.fdtd(scene, frequencies=freqs, excitations=mw.PortSweep(...)).run()

mu = ferrite.permeability_tensor_at_freq(10e9)
result.network.s
```

材料 helper 应支持通过 resonance frequency/linewidth 构造，但内部统一转换为同一组 LLG 参数并把转换结果保存在 metadata，避免两套运行时公式。

## 4. Public API 草案

### 4.1 材料模型

- `GyromagneticFerrite(...)`：`Material` 家族的一等不可变类型；
- 核心参数：`saturation_magnetization [A/m]`、`bias_field [A/m, 3-vector]`、`gilbert_damping >= 0`、`gyromagnetic_ratio > 0`、`eps_r`、`sigma_e`、可选 `mu_infinity`；
- `GyromagneticFerrite.from_resonance(resonance_frequency, linewidth, ...)`：便利构造，仅做有记录的参数转换；
- `polder_tensor(angular_frequency)` / `permeability_tensor_at_freq(frequency)`：解析验证 helper，返回 complex torch tensor；
- 后期 `bias_field` 可接受相对 owning Box 的 `[3,Nx,Ny,Nz]` torch field，但第一阶段只允许均匀 3-vector。

### 4.2 编译与结果

- `Scene.compile_gyromagnetic_materials()`：返回 active-cell parameters、local basis 和 ADE layout；
- `Scene.compile_material_components(frequency=...)` 对 ferrite 返回完整 complex `mu` tensor，而不是丢弃 off-diagonal 项；若现有返回 schema 仅能表达 x/y/z，需新增明确的 full-tensor accessor；
- 内部 `compiler/materials.py::compile_gyromagnetic_layout(...)`，继续由 `Scene.compile_materials(...)` 统一调度；
- `MaterialMonitor` 增加 `mu_xy...`、magnetization 和 magnetic loss 量；普通 `Result`/monitor 访问，不新增 ferrite solver 入口。

单位必须只有一种公共解释。API 不接受含糊的 Oe/Gauss 数值；便利转换放在显式 `from_cgs(...)` helper 并在输出中记录转换。

## 5. 物理与数据模型

### 5.1 频域基准

对局部偏置轴 `z'`，线性响应应给出 Polder 形式

```text
mu_r = [[mu, -i*kappa, 0],
        [i*kappa, mu,  0],
        [0,       0,   mu_parallel]]
```

其中 `mu`、`kappa` 由偏置 Larmor frequency、饱和 magnetization frequency 和阻尼决定。代码必须固定 `exp(-i omega t)` 约定、偏置方向和 gyromagnetic sign；反转 bias 时 `kappa` 变号、非互易方向反转，而对角项不应错误变号。

### 5.2 时域状态

在 ferrite active cells 保存动态横向磁化 `m`。推荐状态合同：

- Faraday curl 推进磁通密度 `B`；
- `B = mu0 * (mu_infinity H + m)`；
- `m` 按线性化 LLG 在静态 bias 附近推进；
- 通过局部坐标基把 `H/B/m` 旋转到 bias frame，执行闭式 2x2/3x3 隐式更新，再旋回全局坐标。

隐式中点或等价 trapezoidal state-space 离散应预计算每 cell 的小矩阵系数。阻尼为零时离散磁能不增长；阻尼为正时功率损耗非负。不得用显式 Euler 依赖极小 `dt` 掩盖不稳定。

### 5.3 编译数据

`CompiledGyromagneticLayout` 使用 active-cell SoA：cell indices、occupancy、bias unit vector、`omega_0`、`omega_m`、damping、`mu_infinity`、预计算 update matrices、magnetization state 和 material-slot ownership。若 active volume 较大，可选择 dense tensor；dispatcher 根据 fill ratio 选择 sparse/dense，但两条路径必须数值 parity，不能一条成为 CPU fallback。

重叠材料遵守当前 Structure priority/occupancy 规则。partial-fill ferrite 在 Phase 1 使用明确的 staircase；conformal/subpixel gyrotropic mixing 只有在有物理推导后才能开启，不能对反对称 tensor 做普通标量线性平均。

## 6. 编译器与原生 CUDA 运行时方案

### 6.1 Compiler

1. 验证参数、bias 非零、resonance 与 simulation band/dt。
2. 计算 geometry occupancy 和 active cell layout。
3. 建立每 cell 正交 local basis；对轴对齐 bias 走无旋转 fast path。
4. 将连续 LLG state-space 用 solver `dt` 离散化，保存 forward 与 reverse 所需系数。
5. 扩展 auto-`dt`/capability metadata；隐式材料不应无理由压缩 CFL，但必须对离散极点/精度给 warning。
6. 生成 material monitor 的完整频域 tensor evaluation。

### 6.2 Forward stepping

磁场更新拆成普通 curl 产生 `B` increment 与 ferrite local correction。CUDA kernel 对 active cell：读取 staggered/共点所需 H/B，旋转到 local frame，更新 `m`，求解新 H，再写回对应 Yee H components。staggered 分量的 collocation/interpolation及其转置必须在设计说明中冻结；不允许简单把三个不同位置的 H 当成天然共点。

优先实现一个 fused `update_gyromagnetic_h`，避免多个 per-component launch。dense 路径按 3D grid，sparse 路径按 active index list。kernel 支持 CUDA graph capture，不在每步创建张量或做 CPU 标量同步。

### 6.3 Reverse stepping

为 local update 写出精确转置，不通过逐步 Python autograd 展开。reverse kernel 传播 H/B/m adjoints，并累计 `omega_0`、`omega_m`、damping、bias direction 和 occupancy/material parameter gradients。checkpoint/replay 保存 magnetization state 或保证从 checkpoint 精确重放。

## 7. PyTorch/autograd 合同

- 均匀 `saturation_magnetization`、bias magnitude/direction、damping、`eps_r`、`sigma_e` 支持 tensor 参数；Phase 1 先交付 scalar parameter gradients，空间 field 后置。
- `permeability_tensor_at_freq` 完全 torch-native，可独立求导并用于拟合材料数据。
- forward FDTD 参数梯度使用自定义 adjoint recurrence；不得 detach 参数后把常数塞入 CUDA。
- bias direction 通过归一化 vector 参数化；零向量拒绝。优化中推荐 spherical/normalized parameter helper，避免尺度与方向退化。
- geometry occupancy 的可微性遵守现有 smooth occupancy 语义；gyrotropic partial-fill 梯度在混合公式被验证前不公开承诺。
- gradient suite 包含反转 bias、接近共振高 Q、低阻尼和复 S 参数 real-valued objective。

## 8. Multi-GPU contract

- collocated ferrite magnetization state `m` 由稳定 global cell id 选择唯一 state owner；Hx/Hy/Hz 仍分别遵守 02 的 component-wise Yee ownership，不能把三个分量写回同一 cell owner。
- 时序固定为 H curl → gather 三个 H/B component 到 state owner → local gyromagnetic constitutive correction → 按离散转置 scatter/reduce 三分量 contribution 到各 component owner → 为后续 E update 交换所需 Hy/Hz halo。
- 普通 halo 只服务 stencil；跨 owner 的 3x3 coupling 使用显式小型 gather/scatter buffer。ghost cell 不保存可写磁化历史；跨界 material monitor/损耗另做 deterministic reduction。
- checkpoint 按 shard 保存 `m`、material slot 和 local coefficients；重分片恢复必须按 global cell id 重映射。
- reverse halo 顺序是 forward 的精确转置，参数梯度使用 deterministic all-reduce。
- 单/多 GPU 的非互易相位、S 参数、能量损耗和梯度均设 parity gate。

## 9. Phases、依赖与 Exit Gates

Phase 0 必须冻结 `AcceptanceBudget`。默认 gate：torch/reference `rtol<=1e-5`；解析 Polder/圆本征模复响应相对误差 `<=2%`、相位误差 `<=3 deg`；被动能量残差 `<=1%`；至少三档 grid/dt/run-length 收敛；支持参数梯度相对误差 `<2%`；multi-GPU 继承 02 parity；无 ferrite 场景性能回退 `<1%`。任何更宽容差必须按高 Q/近共振误差预算预注册。

### Phase 0：物理合同与 torch reference（E0, experimental）

交付物：LLG→Polder 推导、符号/单位约定、隐式离散、离散能量式、单 cell/1D torch reference。

Exit gate：reference 的频率响应在工作带匹配解析 Polder tensor；bias 反转使 off-diagonal 项和 Faraday rotation 反号；阻尼非负时无增益。

### Phase 1：轴对齐均匀偏置 CUDA forward（E1, experimental）

交付物：`GyromagneticFerrite`；z/x/y 轴 bias fast path；compiled active cells；CUDA forward；完整 `mu(f)` evaluator 和 material monitor。

Exit gate：圆极化本征波、均匀 slab transmission/rotation 与解析匹配；CUDA/reference parity；与 PML、conductivity、nonuniform grid 组合运行。

### Phase 2：任意 bias、器件与能量链（E2）

交付物：任意均匀 bias frame；magnetic loss；RF port/network integration；waveguide ferrite slab、isolator/circulator canonical scene。

依赖：RF NetworkData 可用。

Exit gate：bias 旋转协变；非互易器件 `S21/S12` 方向随 bias 反转；输入功率 = 输出 + 反射 + ferrite/其他损耗。

### Phase 3：Adjoint 与可训练材料（E2）

交付物：local reverse kernel；参数梯度；checkpoint/replay；材料拟合示例。

Exit gate：所有连续参数梯度通过高精度有限差分/complex-step 可用项；一个 isolator objective 可反向优化 bias/material 参数且 loss 下降。

### Phase 4：空间 bias、multi-GPU 与性能（E3）

交付物：可选空间 bias field；dense/sparse dispatcher；component-owner gather/scatter 域分解；sharded checkpoint；性能调优。

依赖：02 Phase 5 把 ferrite gather/scatter 加入 advanced-media matrix；空间 multi-GPU gradient 还依赖 02 Phase 7 的 ferrite reverse communication。此前只承诺单 GPU adjoint并明确拒绝 trainable joint solve。

Exit gate：单/多 GPU value/phase/loss/gradient parity；dense/sparse parity；无 ferrite 场景零额外开销，ferrite benchmark 达到批准的 cells/s 与 memory bound。

## 10. 验收策略

### 10.1 单元测试

- SI 参数、CGS 显式转换、bias 非零、damping、resonance helper；
- Polder tensor Hermitian/anti-Hermitian 和 bias reversal 属性；
- state-space 连续/离散 poles；
- local basis 正交与旋转协变；
- sparse/dense index、checkpoint、serialization、compiler cache。

### 10.2 解析测试

- Polder `mu/kappa` 全频曲线；
- bias 方向圆极化 eigen-permeabilities `mu +/- kappa`；
- 无限均匀介质中的 circular birefringence 和 attenuation；
- ferrite slab 的 Faraday rotation、transmission/reflection；
- damping=0 能量守恒、damping>0 单调耗散。

### 10.3 独立求解器/实验

- ferrite-loaded rectangular waveguide 的传播常数与独立 FEM 或公开解析/测量参考；
- 标准三端口 ferrite circulator 的 S 矩阵与独立商业求解器或公开测量；
- 保存材料数据、偏置、参考面、复 S 和网格，不只比较 isolation dB 单点。

### 10.4 收敛、稳定性和性能

- 空间网格、dt、run length、PML distance、material active-cell discretization 收敛；
- resonance 附近长时间稳定和高 Q decay；
- 0/低/典型阻尼下被动性；
- 记录 sparse fill ratio、field update cells/s、active ferrite cells/s、峰值显存和 adjoint 开销；
- 无 ferrite 场景不得增加每步 kernel launch。

## 11. Benchmark 矩阵

| 场景 | 主要指标 | 参考 |
| --- | --- | --- |
| 单 cell/均匀介质 | `mu/kappa`、poles | 解析 Polder |
| 圆极化平面波 | `k+/-`、衰减 | 解析 |
| biased ferrite slab | Faraday rotation、R/T | 解析 transfer matrix |
| ferrite-loaded waveguide | beta、cutoff、loss | 独立 FEM/公开参考 |
| 3-port circulator | S、isolation、insertion loss | 独立求解器/测量 |
| 反转 bias 回归 | `S21 <-> S12` | symmetry relation |
| 多 GPU ferrite volume | parity、scaling | 单 GPU baseline |

场景进入 `benchmark/scenes/ferrite/`，并在 `benchmark/RESULTS.md` 汇总复数误差、能量残差和性能。

## 12. 风险与缓解

- **符号/单位错误仍产生漂亮曲线**：以 Polder tensor、bias reversal 和 circular eigenmode 三重 gate 固定约定。
- **共点化破坏 Yee 能量**：采样与 reverse transpose 一起设计，Phase 0 给出离散能量式。
- **显式材料更新不稳定**：采用局部隐式中点/梯形闭式更新，不用减小 dt 逃避。
- **partial fill 混合错误**：第一版 staircase；经独立推导前不开放 conformal ferrite。
- **高 Q 需要长仿真**：支持窄带 CW/谱累加和 checkpoint，benchmark 明确 run-length convergence。
- **空间 bias 内存过大**：均匀参数去重；空间场只在 active cells 压缩存储。

## 13. 完成定义

1. `GyromagneticFerrite` 从物理参数到 Polder tensor、CUDA forward、Result 和 adjoint 形成单一合同。
2. 任意均匀 bias 的非互易方向正确，反转 bias 可预测地反转器件传输方向。
3. 解析 slab/circular wave、独立 waveguide/circulator、网格/dt 收敛、能量/被动性均通过。
4. 参数梯度、checkpoint 和单/多 GPU value/gradient parity 通过。
5. 无 ferrite 场景零额外状态与 kernel，active-cell 内存线性可解释。
6. 用户示例只使用 `Scene -> Simulation -> Result`，公共能力同步更新 `FEATURE_LIST.md`。

## 14. 修订记录（append-only，不重写历史）

### 2026-07-21 Round-G revision (master `18bc42a`; merge `5dd100e`) — arbitrary-bias forward + mixed-bias support LANDED (not `completed`)

S6 unfroze (S1–S3 passed) and Round G widened the forward beyond the axis-aligned slice.
Evidence per `docs/assessments/g3-ferrite-bias-acceptance-2026-07-21.md`; append-only note.

- **General (non-axis-aligned) bias forward** (`fdtd/runtime/gyromagnetic.py`): the
  axis-aligned guard is lifted; the general path is a pure per-cell coordinate rotation of
  the SAME contracted implicit-midpoint (Cayley) magnetization ADE (frame-independence
  proven in the module docstring) — no new integrator/coefficients; the axis-aligned fast
  path is retained as an exact optimization.
- **Mixed-bias support** (disposition SUPPORT): the "single uniform bias direction" guard is
  removed; different bias axes / opposed signs (`+z`/`−z`) / differing materials route
  through the per-cell general path — the magnetization ADE has no spatial coupling, so a
  mixed-bias scene is the exact direct sum of independent per-cell passive blocks.
- **Gates** (`tests/materials/ferrite/test_gyromagnetic_general_bias.py`; suite 107 passed):
  rotation-equivalence general reduces to fast **bit-for-bit** (max|diff| = 0.0, incl. CUDA);
  oblique-vs-Polder-oracle `chi_uu` rel **1.197e-13**; handedness (bias reversal flips lab
  gyrotropy, co-pol unchanged); Polder spot-check (antisymmetric part flips under reversal);
  mixed-bias per-cell independence (direct-sum bit-for-bit); zero-impact (ferrite-free +
  PEC-only bitwise no-op); CUDA oblique driven-cavity passivity (envelope non-growth 12k
  steps). Falsifications recorded.
- **Contract-doc supersession:** `docs/reference/ferrite-physics-contract.md` §7 item 6
  marked **superseded** (mixed / per-material bias now ships via the per-cell general path);
  historical text retained.
- **Still fail-closed (NOT completed):** Bloch-periodic ferrite, FDFD ingest, multi-GPU,
  adjoint, `PerturbationMedium`-over-ferrite; identity collocation (`C = I`) is not the
  4-point Yee gather (later refinement). Census `175 → 173` (both bias guards removed).
  Measured grade **E1**; no external reference / non-author review, so no `completed`.
