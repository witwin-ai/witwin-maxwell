# 通用表面阻抗与金属粗糙度开发计划

> 状态：proposed  
> 日期：2026-07-14  
> 目标证据：E3 production  
> 类型：现有窄带 SIBC 的正式扩展与替换  
> 前置依赖：现有 `LossyMetalMedium` 原型、PEC/conformal occupancy、ADE/CUDA runtime；RF 端口用于工程验收但不是 Phase 1 前置  
> Owner modules：`media.py`、surface compiler、`fdtd/`、`monitors.py`、`result.py`、Tidy3D adapter  
> 最近架构决策：2026-07-14，统一降低为宽带正实 state-space surface subsystem，迁移后删除旧窄带单平面分叉  
> 公共架构约束：`Scene + Simulation + Result`，GPU-first

## 1. 背景与当前能力

仓库已经实现 `LossyMetalMedium(conductivity=...)` 的第一版 Leontovich surface-impedance boundary：金属内部被 mask，表面切向 E 根据真空侧 H 更新；`Zs` 在单一 operating frequency 上近似为窄带串联 R-L。该实现已有解析 helper 和 CUDA kernel，但编译器明确限制为：

- 单个 axis-aligned `Box` 金属；
- 横跨整个横截面并贴在 domain 一侧，只暴露一个平面；
- normal incidence、实场、非 Bloch；
- 单频窄带 R-L，无任意用户阻抗和粗糙度；
- 不支持有限金属板、中域双面、多金属、边/角、斜面或曲面。

因此本计划不是新增平行的第二种 SIBC，而是把现有 descriptor/runtime 收敛为一个通用、宽带、被动的 surface constitutive subsystem，并让 `LossyMetalMedium` 成为该统一模型的 good-conductor 便利构造。

## 2. 目标与非目标

### 2.1 目标

1. 以因果、被动 rational state-space 表达宽带标量或切向 2x2 表面阻抗/导纳。
2. 支持有限尺寸、中域、多结构、多暴露面，以及 staircased 任意形状；后续支持 conformal 斜面/曲面。
3. 支持 good conductor、用户给定 rational model、频率采样拟合和 Huray/Groisse 类金属粗糙度。
4. 对表面电流、表面损耗和功率平衡提供一等 monitor/Result 数据。
5. 原生 CUDA 推进每个 surface edge 的 ADE state，保持宽带 pulse、Bloch/complex field、checkpoint 和 adjoint 能力。
6. 建立 surface ownership，使域分解下边、角和 partition surface 不重算、不漏算。

### 2.2 非目标

- 不解析金属内部 skin-depth 网格；用户需要内部场时应使用 volumetric conductor。
- 不在第一版支持非局域/spatially dispersive surface、超表面单元间耦合或场强非线性阻抗。
- 不把粗糙几何显式随机生成；Huray/Groisse 是统计等效阻抗模型。
- 不承诺任意频率样本一定能投影为低阶被动模型；拟合失败必须报告误差或拒绝。
- 不在本计划中实现 PCB stackup/CAD 导入。
- 不保留现有单平面 special kernel 作为长期分叉；迁移完成后统一走 surface layout，fast path 只能是同一合同的优化。

## 3. 用户功能描述

### 3.1 Good conductor 与粗糙度

```python
import witwin.maxwell as mw

copper = mw.LossyMetalMedium(
    conductivity=5.8e7,
    frequency_range=(1e9, 40e9),
    roughness=mw.HurayRoughness(
        nodule_radius=0.5e-6,
        surface_ratio=2.0,
    ),
    fit_tolerance=1e-3,
    name="rough_copper",
)
scene.add_structure(mw.Structure(trace_geometry, copper))
```

### 3.2 通用阻抗

```python
surface = mw.RationalSurfaceImpedance(
    poles=poles,
    residues=residues,
    direct=direct,
    frequency_range=(fmin, fmax),
)
coating = mw.SurfaceImpedanceMedium(impedance=surface, name="coating")
scene.add_structure(mw.Structure(panel, coating))
scene.add_monitor(
    mw.SurfaceCurrentMonitor(
        name="panel_current", structure="panel", frequencies=freqs
    )
)
result = mw.Simulation.fdtd(scene, frequencies=freqs).run()
result.monitor("panel_current").surface_current
result.monitor("panel_current").dissipated_power
```

频率采样拟合使用显式工厂，例如 `RationalSurfaceImpedance.fit(frequencies, values, ...)`。它是 prepare-time 工具并返回可检查的 fit report，不在每个 time step 做频域插值。

## 4. Public API 草案

### 4.1 阻抗和粗糙度模型

- `SurfaceImpedanceModel`：只读基类合同，提供 `evaluate(frequency)`、`frequency_range`、passivity/fit metadata。
- `RationalSurfaceImpedance(poles, residues, direct, ...)`：标量或局部切向 2x2 state-space；公共 constructor 严格验证稳定 poles 和被动性。
- `SurfaceImpedanceMedium(impedance, name=None)`：把 surface law 绑定到 `Structure` 外表面。
- `LossyMetalMedium(conductivity, permeability=..., frequency_range=..., roughness=None, ...)`：保留其物理语义，但编译到同一 rational surface model；移除“必须贴 domain 单平面”的公共限制。
- `HurayRoughness(...)`、`GroisseRoughness(...)`：不可变统计参数和有效频带。

粗糙度输入必须对应公开模型的物理参数，不能只给一个无单位“roughness factor”。实际传递函数公式、有效范围和引用写入类 docstring/fit report；超出有效范围返回 warning 或拒绝。

### 4.2 编译和结果

- `Scene.compile_surface_impedances()`：提取 surface topology、局部 basis、material slot、rational coefficients、edge/corner ownership；
- `Scene.compile_surface_monitors()`：解析结构表面选择和频率；
- 内部 `compiler/materials.py::compile_surface_impedance_layout(...)`；现有 `compile_materials` 继续作为统一入口；
- `SurfaceCurrentMonitor` 输出 `J_s [A/m]` 和切向 E/H；表面损耗作为共享 `PowerLossData` 的 surface channel（W/m2 与积分 W）输出，不建立重复 loss monitor/result 类型。
- rational fitting、稳定/被动性 enforcement、连续到离散变换和基础 fit report 复用 03 同一共享内部 rational-model 基础设施；本文只实现切向 surface model、topology 和 surface 专用报告字段。

## 5. 数据模型与物理合同

### 5.1 宽带状态空间

采用局部切向关系

```text
E_t(s) = Z_s(s) J_s(s),       J_s = n x H_t
```

或数值上更稳定的 admittance form。`Z_s(s)` 由正实 rational model 表达：

```text
x_dot = A x + B u
y     = C x + D u
```

其中 `u/y` 根据 impedance/admittance realization 选择。连续稳定模型用 trapezoidal/bilinear 或经过验证的 exact step 离散；更新矩阵在 prepare 时按 `dt` 生成。good conductor 的 `sqrt(s)` 和 roughness correction 在声明频带内拟合到固定阶数，并执行 passivity enforcement。

每个拟合结果保存：原始目标、采样点、阶数、max/RMS complex error、passivity margin、频带和 solver convention。运行频带超出 fit band 必须拒绝或显式重拟合，不能外推后静默运行。

### 5.2 Surface topology

`CompiledSurfaceImpedanceLayout` 保存：

- exposed face ids、结构/material slot 和 outward normal；
- 每个 face 的面积/coverage、局部两个切向 basis；
- face-to-Yee E/H interpolation 与严格转置；
- surface edge/corner adjacency 和唯一 owner；
- 每个 material/state slot 的 rational coefficients；
- 每个 active surface degree of freedom 的 state offset；
- monitor selection、shard ownership 和 geometry fingerprint。

Phase 1 使用从 voxel occupancy 提取的 axis-aligned exposed faces，已经能覆盖有限 block、中域板、多面和 staircase 曲面。Phase 2 才引入基于 cut-face area/normal 的 conformal SIBC；普通 scalar occupancy averaging 不足以表达切向 tensor 和边角电流。

### 5.3 边、角与双面薄板

相邻 faces 在共享 Yee edge 上的贡献按离散表面功率形式组装，不能后写覆盖前写。有限厚 solid 的不同外表面共享 bulk conductor 语义；零厚 sheet 应继续使用 `Medium2D`，除非用户显式选择双面 impedance sheet。编译器必须阻止同一界面同时被 PEC、Medium2D 和 SIBC 重复占有。

## 6. 编译器与原生 CUDA 运行时方案

### 6.1 Surface extraction 与 fitting

1. 从 material occupancy/structure priority 提取 metal-to-nonmetal exposed faces。
2. 合并同 material 的 face batches，建立 edge/corner adjacency 和局部 basis。
3. 根据 simulation source/monitor band 检查或生成 rational fit；pole 数是 compile metadata。
4. 对每个 `dt` 离散 state-space，并执行离散被动性/稳定性检查。
5. 生成 sparse face/edge SoA；高 fill ratio 可生成 dense face masks，但语义相同。
6. 输出 fit/topology report，供 `Simulation.prepare()` 检查。

拟合器可在 CPU prepare-time 使用线性代数，但它不是 time-stepping fallback；最终 coefficients 一次性上传 GPU。若参数是可训练张量，采用固定 pole basis 的 torch solve 或直接要求用户传 rational coefficients，不能经 NumPy detach 后声称可微。

### 6.2 CUDA forward

统一 kernel 序列：采样切向 H/E → 更新 surface ADE states → 计算边界切向场/电流 → 以 transpose weights 写回 E update → 累积可选损耗。针对标量 good conductor 可有 fused fast path，但必须与 generic state-space parity。

kernel 支持：

- real 和 Bloch complex field；
- 任意 axis-aligned face orientation；
- 多 material/state slot；
- surface edges/corners 的 deterministic segmented reduction；
- CUDA graph capture、checkpoint/restart；
- 没有 SIBC 时零 launch 和零 state。

Phase 2 conformal kernel读取 face area、normal 和切向 2x2 coupling；曲面只是 topology/layout 差异，不再新增第三套 solver。

### 6.3 Reverse

对 state-space recurrence、field interpolation、edge reduction 写精确转置 CUDA kernel。surface state 纳入 reverse checkpoint/replay；损耗 monitor seeds 可以回传到 conductivity、roughness 和 rational coefficients。

## 7. PyTorch/autograd 合同

- 直接传入的 rational coefficients、surface monitor 后处理和 material helper 完全 torch-native。
- 固定 pole basis 下，residue/direct、conductivity 和连续 roughness 参数可微；自动选阶、pole relocation 和 passivity projection 是离散/分段操作，不承诺梯度。
- `fit(...)` 返回的 report 标明 `differentiable_parameters`。若用户需要优化，推荐冻结 poles，仅优化 residues/物理参数并在每步检查 passivity。
- conformal geometry gradient 仅在 topology 不改变的固定 cut-face stencil 内有效；face 出现/消失是非光滑事件。
- reverse 测试覆盖复阻抗、宽带 pulse、粗糙度参数和表面损耗 objective。

## 8. Multi-GPU contract

- surface degree of freedom 由相邻 nonmetal E edge owner rank 独占；共享 face/edge 使用 global face id 和稳定 tie-break，避免重复 state。
- 更新前读取完成 halo 的切向 H；写入非 owner E edge的贡献进入邻居 reduction buffer。
- ADE state、fit slot 和 monitor partials 随 surface owner 分片；checkpoint 以 global face/edge id 可重分片恢复。
- surface dissipated power 和 current integral 做 deterministic all-reduce；SIBC 子系统不计算 S 参数，01 的 `NetworkData` 功率审计消费归约后的 surface loss。
- reverse 通信是 forward interpolation/deposition 的精确转置，参数梯度 all-reduce。
- partition 穿过金属表面、边、角、双面板的单/多 GPU parity 是必测项。

## 9. Phases、依赖与 Exit Gates

Phase 0 必须冻结 `AcceptanceBudget`。默认 gate：rational fit max complex error `<=1e-3` 或用户更严阈值；解析/独立参考复反射或损耗相对误差 `<=2%`、相位误差 `<=3 deg`；功率残差 `<=1%` 且局部耗散非负；至少三档 grid/dt/fit-order 收敛；支持参数梯度相对误差 `<2%`；multi-GPU 继承 02 parity；无 SIBC 场景性能回退 `<1%`。定性“通过/一致”均引用该预算。

### Phase 0：统一合同和宽带 reference（E0, experimental）

交付物：surface state-space schema、正实拟合/被动性报告、离散能量式；把现有窄带平面案例复现到 generic torch reference。

Exit gate：generic model 在单频退化情况下复现现有 `LossyMetalMedium` 解析反射；宽带 good-conductor `Zs` 在声明频带达到 fit tolerance；离散 poles 稳定且被动。

### Phase 1：通用 staircased 宽带 SIBC（E2）

交付物：统一 material compiler/layout；有限 block、中域双面、多金属、多朝向；generic CUDA ADE；Bloch/complex；surface current/loss monitor。

Exit gate：平面宽带反射、有限 PEC/metal plate、box 多面与多个结构通过；现有单平面 special restriction 删除；能量平衡闭合。

### Phase 2：Conformal 斜面与曲面（E2）

交付物：cut-face area/normal、局部 2x2 tangential coupling、edge/corner assembly、曲面 monitor mapping。

依赖：现有 conformal PEC occupancy 的几何信息可复用，但必须增加 surface-specific energy derivation。

Exit gate：旋转平板响应具有网格收敛的旋转不变性；有限 conductivity sphere/cylinder scattering 与解析/Mie 或独立求解器一致；无负耗散。

### Phase 3：Huray/Groisse 粗糙度（E2）

交付物：两个一等 roughness 模型、频带/参数校验、与 good-conductor impedance 组合、拟合报告、表面损耗分解。

Exit gate：模型实现逐频匹配公开公式；粗糙度为零精确退化到 smooth metal；microstrip/stripline insertion loss 与独立求解器或测量趋势和幅值一致。

### Phase 4：Adjoint、multi-GPU 与性能（E3）

交付物：surface reverse kernel；物理/rational 参数梯度；sharded state/checkpoint；dense/sparse/fused dispatcher。

依赖：02 Phase 5 的 surface owner/reduction 和 Phase 7 的 surface reverse communication。此前只承诺单 GPU adjoint，并对 trainable joint solve 设 prepare-time guard。

Exit gate：参数梯度有限差分通过；单/多 GPU value/loss/gradient parity；无 SIBC 零开销；surface-heavy benchmark 达到批准的 memory/throughput 指标。

## 10. 验收策略

### 10.1 单元测试

- stable poles、共轭配对、正实/被动性、fit band 和误差 report；
- exposed face 提取、normal、finite plate 双面、edge/corner ownership；
- impedance/admittance realization parity；
- compiler cache、serialization、checkpoint、Bloch complex state；
- `Medium2D`/PEC/SIBC overlap rejection。

### 10.2 解析测试

- smooth good conductor 的 `Zs=(1-i)sqrt(omega*mu/(2*sigma))`；
- 任意入射角 TE/TM Leontovich reflection；
- rational R、L、C surface 的 impulse/frequency response；
- conductor half-space absorption 与表面功率积分；
- roughness=0 与低/高频渐近行为。

### 10.3 独立求解器/测量

- finite conductivity plate/sphere/cylinder scattering 与 Mie/独立 FEM；
- copper microstrip/stripline/cavity Q 的 smooth/rough conductor loss，与独立商业求解器；
- 至少一个带 profilometer roughness 参数和 VNA insertion-loss 的公开测量案例。

### 10.4 收敛、能量与性能

- 网格、dt、fit order、fit samples、frequency band、conformal surface refinement 收敛；
- 每频点输入 = 反射 + 透射/辐射 + surface/material loss；每个 surface cell 的平均耗散不得为负；
- 比较 resolved skin-depth volumetric conductor 的重叠有效区；
- 记录 active surface DoF、state order、峰值显存、每步时间、fit time；内存应随 surface area × order，而非 metal volume × order 增长。

## 11. Benchmark 矩阵

| 场景 | 主要指标 | 参考 |
| --- | --- | --- |
| good-conductor half-space | 宽带 complex R、absorption | 解析 Fresnel/SIBC |
| 斜入射平板 | TE/TM、角度 sweep | 解析 Leontovich |
| finite metal plate/box | edge diffraction、loss | 独立 FEM/FDTD |
| conducting sphere/cylinder | scattering、absorption | Mie/解析 |
| smooth microstrip | attenuation、S21 | transmission-line theory + FEM |
| rough microstrip/stripline | insertion loss、phase | Huray/Groisse reference + 测量 |
| cavity | Q、surface loss map | 解析/测量 |
| 跨 GPU 曲面 | parity、scaling | 单 GPU baseline |

场景进入 `benchmark/scenes/surface_impedance/`，拟合目标和原始复数结果作为 cache artifact 保存并汇总到 `benchmark/RESULTS.md`。

## 12. 风险与缓解

- **rational fit 准确但非被动**：正实检查与 passivity enforcement 是编译 exit gate，不能作为 warning 跳过。
- **边角多面重复更新**：显式 topology/owner 和功率形式组装，不用按 face 顺序覆盖 E。
- **conformal normal/collocation 破坏稳定**：Phase 2 前先给离散能量推导，并做旋转/能量 benchmark。
- **粗糙度模型超出有效范围**：类保存模型有效频带/参数限制，fit report 披露 extrapolation。
- **fit 与 autograd 冲突**：区分离散 fitting 和固定-pole differentiable evaluation；API 不虚假承诺自动选阶可微。
- **现有原型形成双路径负担**：Phase 1 以 generic layout 复现后删除旧 special dispatch；fast path 必须共享数据合同和 parity tests。

## 13. 完成定义

1. `LossyMetalMedium` 与通用 `SurfaceImpedanceMedium` 统一编译为宽带被动 state-space，不再受单贴边平面限制。
2. 有限/多面/多金属/staircase 与 conformal 斜曲面均能输出表面电流和非负耗散。
3. Huray/Groisse 参数、公式、频带和 fit error 可审计，roughness=0 精确退化。
4. 解析、Mie/FEM/测量、resolved-skin-depth 重叠区、收敛、能量和性能验收通过。
5. adjoint 与 single/multi-GPU value/loss/gradient parity 通过，无 SIBC 场景零开销。
6. 公共示例遵守 `Scene -> Simulation -> Result`，`FEATURE_LIST.md` 同步记录能力和有效范围。

## 14. 修订记录（append-only，不重写历史）

### 2026-07-21 Round-G revision (master `589188e`; merge `ac3719b`) — all-orientation staircased SIBC + cylinder physics gate + skin-effect bench LANDED (not `completed`)

S6 unfroze (S1–S3 passed) and Round G generalized the boundary beyond the axis-aligned
exposed-face layout. Evidence per `docs/assessments/g4-sibc-oblique-acceptance-2026-07-21.md`;
append-only note.

- **All-orientation staircased exposed-face SIBC** (`compiler/materials.py`,
  `fdtd/runtime/materials.py`, `fdtd/runtime/stepping.py`): a non-`Box` good conductor is
  staircased from its node occupancy — each axis-aligned voxel face becomes a masked
  Leontovich surface-impedance write (cylinder/sphere, all six orientations, mixed).
  Orientation-equivalence (cyclic-permutation residual ~1.7e-7) + mixed-orientation stability
  gates (`test_sibc_orientation.py` 6, `test_sibc_staircase.py` 6); staircase analytic
  flat-plate match <1%. Falsifications: face-normal sign flip + active-branch sign flip.
- **Staircased-cylinder physics convergence gate** (`test_sibc_cylinder_convergence.py`, 5):
  SIBC/resolved absorbed-power ratio ≈ **0.18**, **grid- AND R/δ-independent** (0.172 at
  R/δ=6.7, 0.194 at R/δ=10.1) — the intrinsic **first-order-Leontovich-on-a-staircased-curve
  systematic** (flat-surface value <1%); a half-cell low/high surface-node placement asymmetry
  is a documented contributing convention (not a bug). Gate at 25% (fails closed toward PEC).
  Committed probes under `docs/assessments/g4-sibc-oblique-probes/`.
- **Wave-level skin-effect attenuation benchmark**
  (`benchmark/scenes/rf/lossy_waveguide_attenuation.py`): alpha vs analytic `alpha_c`
  (Pozar 3.96) median rel err **0.049%** / max 0.37% (<5%); PEC-wall falsification collapses
  alpha to 0.00017 Np/m. The one authorized external-reference run is recorded **honestly** —
  the external lossy-metal surface-impedance export **under-applies the wall loss** (a
  documented adapter-fidelity gap, NOT the FDTD bench, which matches analytic to 0.05%).
- **Committed zero-impact gate** (`test_sibc_zero_impact.py`): a SIBC-free scene's six raw
  Yee fields are **bitwise identical** with the compile hook present vs monkeypatched out,
  with a load-bearing control; falsification recorded.
- **Still fail-closed (NOT completed):** true oblique/conformal (non-staircase) SIBC, rotated
  `Box`, generic rational (`SurfaceImpedanceMedium`) ADE on a curved conductor, Bloch + SIBC,
  adjoint/distributed/trainable SIBC, generic surface-impedance adapter export. The ~18%
  curved-conductor systematic awaits a curvature-corrected surface impedance. Census unchanged
  (SIBC funnel narrowed, no guard added/removed). Measured grade **E1**; the external
  cross-check is a documented adapter gap (not a pass) and there is no non-author review, so
  no `completed`.
