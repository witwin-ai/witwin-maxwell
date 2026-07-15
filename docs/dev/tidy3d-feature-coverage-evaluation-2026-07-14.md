# Tidy3D Feature-Coverage 完整数值评估报告

日期：2026-07-14（America/Los_Angeles）  
评估对象：`witwin.maxwell` 与 Tidy3D 的端到端数值一致性  
基础提交：`a8e7927574889fe5ee3a111d1c48a5ff2e4894f8`（`master`，评估在包含未提交开发改动的工作区上执行）

## 1. 执行摘要

本报告仅包含 101 个以 Tidy3D 为 reference 的 benchmark 场景，其中包括 53 个专门的 `feature_coverage` 场景。场景注册、Tidy3D reference、Maxwell 求解、监视器提取、标量后处理、绘图和 Markdown 汇总链路均已实际执行。任何本地求解器互相比对均排除在报告之外。

核心结论：

1. **覆盖完整性已经达到当前直接可比范围内的目标。** 53/53 个 `feature_coverage` 场景均有真实 public object、已注册、可构建、具有 reference，并完成 Maxwell-vs-reference 计算。
2. **数值一致性尚未全面达标。** 101 个 Tidy3D 对比场景中只有 41 个满足 `Field L2 < 0.1`，54 个满足 `Field Corr > 0.99`；53 个 feature-coverage 场景中相应为 19 个和 28 个。
3. **新增覆盖是有效的。** 它没有通过弱断言制造“全绿”结论，而是暴露了 higher-order mode、TFSF/Bloch 幅值归一化、RCS、时间轨迹、复杂几何和非线性/调制材料等真实薄弱区。
4. **当前最优先的问题并不全是求解器误差。** 多个场景具有很好的 Shape L2/Corr，但原始 L2 或 flux 极大，说明比较、源幅值、功率归一化或后处理尺度链路需要先于底层数值内核排查。

原始结果与全部图像：

- [benchmark/RESULTS.md](../../benchmark/RESULTS.md)
- [benchmark/plots/](../../benchmark/plots/)
- [FEATURE_LIST.md](../../FEATURE_LIST.md)

## 2. 评估范围

### 2.1 场景与结果完整性

| 项目 | 数量 | 状态 |
|---|---:|---|
| Tidy3D 对比场景 | 101 | 全部完成 |
| `feature_coverage` 场景 | 53 | 全部完成 |
| Tidy3D validation catalog 条目 | 87 | 注册检查通过 |
| 本轮实际使用的 Tidy3D cache hit | 101 | 全部命中，无云端重跑 |
| 主结果行 | 101 | 完整 |
| 逐频率场指标行 | 171 | 与全部 Tidy3D 场景频点总数一致 |
| scalar observable 行 | 155 | 无未知或过期场景行 |
| Tidy3D 场景图像 | 320 | 完整 |

### 2.2 覆盖的功能族

本轮 feature-coverage 包括：

- Sources：Gaussian beam、CW、TFSF、higher-order ModeSource、磁流源、Ricker、负方向传播和各向异性极化。
- Media：Drude、Lorentz、多极点、Debye、Sellmeier、对角各向异性、PEC、LossyMetal、Kerr、调制、静态 Medium2D、MaterialRegion、色散与非线性组合。
- Boundaries：PML、材料加载 PML、periodic、Bloch、PEC/PMC symmetry、混合 face、非对称 face。
- Grid / Geometry：解析曲面、Mesh、PolySlab、AutoGrid、custom grid、各向异性 uniform grid、override 和 layer refinement。
- Postprocess / Monitors：RCS、directivity、ModeMonitor、ModePort、diffraction、PointMonitor、PermittivityMonitor、FieldTimeMonitor、FluxTimeMonitor。

不具备直接 Tidy3D 等价语义的能力不计入“直接可比覆盖”，而是在 adapter 中明确拒绝，例如空间变化或过滤后的 `MaterialRegion`、非单位 `mu_r` 的 `MaterialRegion`，以及无法等价映射到 Tidy3D 中心对称面的局部 face-symmetry 场景。

## 3. 执行环境与复现命令

| 组件 | 版本/信息 |
|---|---|
| Python | 3.11.14 |
| PyTorch | 2.10.0 |
| PyTorch CUDA runtime | 12.8 |
| Tidy3D | 2.10.2 |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB |
| NVIDIA driver | 596.49 |
| Conda environment | `witwin2` |

Reference 补全命令：

```powershell
conda run -n witwin2 python -m benchmark --references-only
```

完整 Maxwell-vs-reference 评估命令：

```powershell
conda run -n witwin2 python -m benchmark --solver fdtd
```

最终回归命令：

```powershell
conda run -n witwin2 python -m pytest tests/api/adapters/tidy3d tests/validation/benchmark -q
```

最终结果：`223 passed`。

## 4. 指标定义与判定方式

| 指标 | 含义 | 本报告筛查线 |
|---|---|---:|
| Field L2 | 原始复场相对 L2，包含幅值和全局相位差异 | `< 0.1` |
| Shape L2 | 去除最佳拟合全局复尺度后的场型 L2 | `< 0.1`（诊断线） |
| Field Linf | 最大局部相对偏差 | `< 0.1` |
| Field Corr | 归一化复相关性 | `> 0.99` |
| Flux err | incident-power-normalized flux error | `< 0.05` |
| Complex err | 标量复数观测量的对称相对误差 | 越小越好 |
| Phase err | 包裹后的相位误差 | 越小越好 |

Shape L2 用于区分两类问题：

- Field L2 很大、Shape L2 很小：更可能是幅值、单位、源归一化或全局相位约定问题。
- Field L2 和 Shape L2 都大、Corr 低：更可能是传播、边界、材料、模式或几何离散本身不一致。

上述阈值用于工程筛查，并不等同于正式科学验证标准。不同物理量和应用仍应设置各自容差。

## 5. 全局统计

| 范围 | Field L2 < 0.1 | Shape L2 < 0.1 | Linf < 0.1 | Corr > 0.99 | Flux < 0.05 |
|---|---:|---:|---:|---:|---:|
| 101 个 Tidy3D 场景 | 41/101（40.6%） | 44/101（43.6%） | 30/101（29.7%） | 54/101（53.5%） | 47/86（54.7%） |
| 53 个 feature-coverage | 19/53（35.8%） | 22/53（41.5%） | 14/53（26.4%） | 28/53（52.8%） | 27/46（58.7%） |

这组数据说明：当前系统已经具备较宽的功能通路，但不能把“可运行、可导出、可比较”视作“数值等价已经验证”。

## 6. Feature-Coverage 分族统计

| 功能族 | 场景数 | Shape < 0.1 | Corr > 0.99 | Flux < 0.05 | Shape L2 中位数 |
|---|---:|---:|---:|---:|---:|
| Sources | 9 | 5 | 6 | 3/9 | 0.0272 |
| Media | 14 | 6 | 8 | 9/13 | 0.1261 |
| Boundaries | 8 | 5 | 5 | 2/8 | 0.0514 |
| Grid / Geometry | 12 | 2 | 4 | 11/12 | 0.1759 |
| Postprocess | 10 | 4 | 5 | 2/4 | 0.1414 |

主要观察：

- Sources 的场型中位数最好，但 TFSF、higher-order mode 和磁流源形成明显长尾。
- Boundaries 的典型场型较好，但 periodic、Bloch flux 和 mixed faces 需要重点处理。
- Grid / Geometry 只有 2/12 达到 Shape L2 诊断线，不过 11/12 的 flux 达标，说明主要矛盾集中在局部界面/场分布，而非全局功率守恒。
- Postprocess 中 directivity、diffraction 和 mode ratio 有可用结果，但 RCS 和时间轨迹仍不可靠。

## 7. 53 个 Feature-Coverage 场景明细

### 7.1 Sources

| Scenario | Field L2 | Shape L2 | Corr | Flux err |
|---|---:|---:|---:|---:|
| `gaussian_beam_normal` | 1.1598e-02 | 1.1450e-02 | 0.9999 | 1.9017e-03 |
| `gaussian_beam_defocused` | 2.7782e-02 | 2.7220e-02 | 0.9996 | 1.7866e-03 |
| `planewave_cw` | 2.9859e-02 | 1.7255e-02 | 0.9999 | 5.6097e-02 |
| `dipole_cw_vacuum` | 3.6482e-03 | 2.6766e-03 | 1.0000 | 3.5671e-03 |
| `tfsf_vacuum` | 2.5365e+07 | 1.5399e-02 | 0.9999 | 2.2690e+14 |
| `tfsf_dielectric_sphere` | 2.4524e+07 | 4.7129e-01 | 0.8820 | 6.6159e+14 |
| `mode_source_higher_order` | 5.7369e+02 | 9.9997e-01 | 0.0080 | 9.4459e-01 |
| `magnetic_current_vacuum` | 3.9451e-01 | 3.6934e-01 | 0.9293 | 2.1716e-01 |
| `ricker_axis_x_anisotropic` | 1.3047e-01 | 1.2580e-01 | 0.9921 | 3.2959e-01 |

### 7.2 Media

| Scenario | Field L2 | Shape L2 | Corr | Flux err |
|---|---:|---:|---:|---:|
| `drude_slab` | 2.1079e-02 | 2.0480e-02 | 0.9998 | 2.5383e-02 |
| `lorentz_slab` | 1.1381e-01 | 1.1368e-01 | 0.9935 | 3.5293e-02 |
| `lorentz_two_pole_slab` | 1.4902e-01 | 1.4901e-01 | 0.9888 | 2.3769e-02 |
| `drude_lorentz_slab` | 8.8844e-02 | 8.8800e-02 | 0.9960 | 2.9409e-02 |
| `debye_sphere` | 4.2868e-02 | 4.2641e-02 | 0.9991 | 1.3084e-02 |
| `sellmeier_sphere` | 3.0362e-01 | 2.9476e-01 | 0.9556 | 1.1736e-02 |
| `diag_aniso_sphere` | 7.6616e-02 | 7.5861e-02 | 0.9971 | 6.6983e-03 |
| `pec_sphere` | 3.4255e-01 | 3.4040e-01 | 0.9403 | 2.7019e-02 |
| `lossy_metal_slab_high_sigma` | 2.0440e-02 | 1.3645e-02 | 0.9999 | 3.9433e-02 |
| `kerr_slab_strong` | 1.4146e-01 | 1.3845e-01 | 0.9904 | 1.6381e+00 |
| `modulated_slab_phase` | 3.5136e-01 | 3.3457e-01 | 0.9424 | - |
| `static_medium2d_sheet` | 4.8001e-02 | 4.7307e-02 | 0.9989 | 6.7258e-02 |
| `material_region_slab` | 2.2325e-01 | 2.2315e-01 | 0.9748 | 1.3723e-01 |
| `dispersive_kerr_slab` | 4.0085e-01 | 3.7835e-01 | 0.9257 | 3.4569e-01 |

### 7.3 Boundaries

| Scenario | Field L2 | Shape L2 | Corr | Flux err |
|---|---:|---:|---:|---:|
| `pml_thin` | 7.4121e-02 | 7.4095e-02 | 0.9973 | 2.2100e-02 |
| `pml_slab_through` | 3.4497e-02 | 2.8608e-02 | 0.9996 | 1.0867e-01 |
| `periodic_slab` | 3.1148e-01 | 3.0953e-01 | 0.9509 | 2.0088e-01 |
| `bloch_oblique_te` | 2.5924e+07 | 1.9059e-01 | 0.9817 | 6.2078e+14 |
| `symmetry_pec_center` | 6.5408e-02 | 1.3164e-02 | 0.9999 | 1.2746e-01 |
| `symmetry_pmc_center` | 6.5448e-02 | 1.3127e-02 | 0.9999 | 1.2824e-01 |
| `mixed_faces` | 3.4488e-01 | 3.4227e-01 | 0.9396 | 2.8187e+00 |
| `asymmetric_boundary_faces` | 1.3173e-02 | 1.3164e-02 | 0.9999 | 7.6607e-03 |

### 7.4 Grid / Geometry

| Scenario | Field L2 | Shape L2 | Corr | Flux err |
|---|---:|---:|---:|---:|
| `cylinder_scatter` | 2.3596e-01 | 2.3461e-01 | 0.9721 | 1.1151e-02 |
| `cone_scatter` | 5.9606e-01 | 5.6047e-01 | 0.8282 | 6.9713e-02 |
| `ellipsoid_scatter` | 1.2451e-01 | 1.2425e-01 | 0.9923 | 6.0322e-03 |
| `pyramid_scatter` | 4.4884e-01 | 4.3969e-01 | 0.8982 | 6.8926e-03 |
| `prism_scatter` | 1.8923e-01 | 1.8850e-01 | 0.9821 | 1.0966e-02 |
| `hollow_box_scatter` | 5.2624e-01 | 5.0973e-01 | 0.8603 | 2.7973e-02 |
| `polyslab_pentagon` | 1.6368e-01 | 1.6338e-01 | 0.9866 | 1.6794e-02 |
| `autogrid_slab` | 3.8830e-02 | 3.8728e-02 | 0.9992 | 1.7206e-02 |
| `nonuniform_custom_grid` | 1.3829e-01 | 1.3773e-01 | 0.9905 | 3.7704e-02 |
| `anisotropic_uniform_grid` | 9.5548e-02 | 9.5489e-02 | 0.9954 | 4.4381e-02 |
| `explicit_mesh_scatter` | 4.4884e-01 | 4.3969e-01 | 0.8982 | 6.8926e-03 |
| `autogrid_override_refinement` | 1.5905e-01 | 1.5886e-01 | 0.9873 | 7.0740e-03 |

### 7.5 Postprocess / Monitors

| Scenario | Field L2 | Shape L2 | Corr | Flux err |
|---|---:|---:|---:|---:|
| `rcs_pec_sphere` | 3.0437e+07 | 1.8223e-01 | 0.9833 | - |
| `rcs_dielectric_box` | 2.8447e+07 | 1.7457e-01 | 0.9846 | - |
| `directivity_two_dipoles` | 7.4083e+07 | 5.4014e-02 | 0.9985 | - |
| `mode_monitor_straight_wg` | 5.3432e-01 | 1.6048e-01 | 0.9870 | - |
| `mode_monitor_two_planes` | 5.3432e-01 | 1.6048e-01 | 0.9870 | - |
| `diffraction_normal_orders` | 8.1067e-02 | 8.1067e-02 | 0.9967 | 6.3000e-03 |
| `point_monitor_probe` | 1.7377e-03 | 1.7246e-03 | 1.0000 | 1.1317e-01 |
| `mode_port_straight_wg` | 1.4726e+00 | 1.7954e-01 | 0.9837 | - |
| `permittivity_monitor_slab` | 1.2310e-01 | 1.2224e-01 | 0.9925 | 1.2601e-02 |
| `time_monitor_vacuum` | 1.5956e+08 | 9.8153e-02 | 0.9952 | 6.3966e-01 |

## 8. 关键标量观测量

| 场景/观测量 | 结果 | 评价 |
|---|---:|---|
| `mode_monitor_two_planes` forward amplitude ratio | complex error 0.00264，phase 0.00260 rad | 很好 |
| `mode_port_straight_wg` forward amplitude ratio | complex error 0.0535，phase 0.0126 rad | 可用，但场分布仍有偏差 |
| ModeMonitor / ModePort effective index | error 0.0549 | 中等，需要校准模式求解 |
| `permittivity_monitor_slab` mean eps | 2.85%–3.09% | 可用 |
| `permittivity_monitor_slab` min/max eps | 0 | 完全一致 |
| `diffraction_normal_orders` ±1 order | 1.01% | 很好 |
| `diffraction_normal_orders` 0th order | 4.73% | 可用 |
| `directivity_two_dipoles` Dmax | 3.29% | 可用 |
| `directivity_two_dipoles` beam widths | 0.88%–1.80% | 很好 |
| 两个 RCS 场景的四个方向 | error 1.0 | 不可用 |
| `time_monitor_vacuum` field trace | waveform L2 1.198 | 不可用 |
| `time_monitor_vacuum` flux trace | waveform L2 1.234 | 不可用 |

PointMonitor 的主平面场型非常好，但多个弱横向 probe 分量出现 0.84–1.01 的相对误差。需要检查这些分量是否接近零；若是，应使用带绝对误差下限或相对 reference 主分量的稳定指标，避免近零分母把噪声放大成失败。

## 9. 表现可靠的区域

以下路径已经表现出较强一致性，可作为回归基线：

- 真空 dipole、不同 dipole 极化和 CW dipole。
- Gaussian beam 正常聚焦与离焦。
- PML vacuum、材料加载 PML 的场型。
- PEC/PMC symmetry 和非对称 face boundary 的场型。
- Drude、Debye、对角各向异性、LossyMetal、静态 Medium2D。
- AutoGrid 基础 slab 与各向异性 uniform grid。
- diffraction 场型及主要衍射阶效率。
- directivity 的方向性标量。
- 两平面模式传播比和 ModePort 传输比。
- PermittivityMonitor 的材料统计。

其中 `tfsf_vacuum` 和 `directivity_two_dipoles` 的场型也很好，但原始幅值尺度错误，因此只能作为 shape 回归基线，不能作为绝对场强或功率基线。

## 10. 主要问题与优先级

下表中的“可能层级”是由指标形态推断出的排查方向，不是最终根因结论。

| 优先级 | 问题族 | 代表证据 | 可能层级 | 建议动作 |
|---|---|---|---|---|
| P0 | TFSF 幅值/功率归一化 | vacuum Shape 0.0154、Corr 0.9999，但 Field L2 2.54e7、flux 2.27e14 | source amplitude / SI-to-Tidy units / incident power | 对单一平面波解析幅值逐层核对 source、monitor、cache rescale 和 flux normalization |
| P0 | higher-order ModeSource | Shape 0.99997、Corr 0.008 | mode ordering / polarization sorting / launch field | 对 Maxwell/Tidy 模式剖面、传播方向、归一化和 mode index 做逐层可视化 |
| P0 | RCS | 四个方向 scalar error 均为 1.0 | closed-surface field units / near-to-far scale | 在公共 near-to-far 输入前比较六个切向场与 SI 单位，先验证解析 PEC 小球量纲 |
| P0 | 时间监视器 | 场型 Corr 0.9952，但 trace L2 1.198/1.234 | time origin / sign / interpolation / source normalization | 保存实际时间坐标，使用物理时间插值和允许延迟的互相关诊断 |
| P0 | Bloch TE flux | Field L2 2.59e7、flux 6.21e14、Shape 0.191 | oblique source normalization / Bloch power | 分离 Bloch phase、source sheet amplitude 与斜入射法向功率 |
| P1 | guided/modal 系统 | ring Shape 0.9977、waveguide S-matrix Shape 0.9423 | eigenmode field / port convention / propagation constant | 建立直波导单模解析基线，再进入 ring 和多端口 |
| P1 | 复杂几何 | cone 0.560、hollow box 0.510、mesh/pyramid 0.440 Shape L2 | voxelization / subpixel / surface placement | 输出两侧材料 mask、界面距离和 Yee 分量采样位置；进行网格收敛 |
| P1 | full tensor / modulation / nonlinear | full tensor Shape 0.644；modulated 0.269–0.335；dispersive Kerr 0.378 | constitutive update / tensor rotation / nonlinear convention | 分解为静态线性、单轴 tensor、弱非线性、单边带四级测试 |
| P1 | PMC cavity | Shape 0.712、Corr 0.702 | PMC staggering / modal spectrum / termination | 与解析腔模和 PEC 对偶场逐分量比较 |
| P2 | MaterialRegion | Shape 0.223、flux 0.137 | region interpolation / boundary occupancy | 比较编译后的 eps tensor 与导出的均匀结构占据范围 |
| P2 | periodic/mixed faces | periodic Shape 0.310；mixed Shape 0.342、flux 2.82 | face staggering / monitor aperture / boundary interaction | 先做纯 vacuum periodic，再逐 face 增加 PEC/PML |
| P2 | magnetic current | Shape 0.369、Corr 0.929 | magnetic-source sign/unit convention | 用电磁对偶的解析 dipole 对比 J/M 映射 |

## 11. 建议的修复阶段

### 阶段 A：先修比较与量纲链路

目标：避免把单位/归一化错误误判为求解器误差。

1. TFSF、Bloch、RCS、directivity 和 time monitors 增加原始值、缩放前值、缩放后值的诊断记录。
2. 所有功率指标记录 aperture、normal、SI 单位、source reference amplitude 和 normalization denominator。
3. 时间监视器保留实际时间坐标，不只保留样本数组。
4. 近零 scalar 使用绝对误差门槛或相对于主量的误差。

### 阶段 B：模式系统

1. 建立单一均匀直波导的 mode field、n_eff、forward/backward amplitude 基线。
2. 修复 higher-order mode 后再评估 ModePort、S-matrix 和 ring。

### 阶段 C：复杂几何与材料

1. 对 cylinder/cone/pyramid/mesh 做 2–3 级网格收敛。
2. 比较 Maxwell 编译 material mask 与 Tidy3D 几何截面。
3. 分离 full tensor、时间调制、Kerr 和色散组合，避免多个物理机制同时进入失败场景。

## 12. 性能摘要

| 指标 | 结果 |
|---|---:|
| Tidy3D 对比场景 Maxwell 求解器累计时间 | 403.94 s |
| FDTD 中位 ms/step | 0.5296 ms |
| FDTD ms/step 范围 | 0.3104–8.1701 ms |
| FDTD 最大增量显存 | 644 MiB（`mode_source_wg`） |

最慢的求解器场景包括：

- `ring_resonator_s21`：43.34 s。
- `pmc_cavity`：32.07 s。
- `pec_cavity`：31.79 s。
- `sellmeier_slab`：20.24 s。

墙钟时间包含模式规划、数据提取、插值、Matplotlib 绘图和 Markdown 写入，因此高于求解器累计时间。

## 13. 评估过程中发现并修复的基础设施问题

本轮完整运行本身发现了三个会妨碍真实评估的问题：

1. `ModePort` 场景没有直接 `scene.sources`，绘图代码错误地访问 `scene.sources[0]`。现已统一从 `resolved_sources()` 获取。
2. 时间轨迹重采样为 128 点后被通用 scalar 比较器错误地要求“长度等于频率数量”。现已按完整归一化波形计算 L2。
3. benchmark 原先只在整批结束后写 `RESULTS.md`。现已按场景增量持久化，并正确合并主结果、scalar 和逐频率行；重跑场景时会清除该场景的旧行，避免残留数据。

相关实现：

- [benchmark/plotting.py](../../benchmark/plotting.py)
- [benchmark/runner.py](../../benchmark/runner.py)
- [benchmark/report.py](../../benchmark/report.py)
- [tests/validation/benchmark/test_benchmark_system.py](../../tests/validation/benchmark/test_benchmark_system.py)

## 14. 限制与版本管理注意事项

1. 本报告评估的是包含未提交修改的工作区状态，不是纯净的基础提交 `a8e7927`。
2. 当前 `.gitignore` 忽略整个 `benchmark/` 目录，因此场景代码、reference cache、图像、`RESULTS.md`、runner/report/plotting 修改不会出现在普通 `git status` 中。若需要提交或在 CI 中复现，必须先明确调整跟踪策略。
3. Tidy3D reference 是云端求解结果；本轮完整 Maxwell 评估全部复用缓存，没有再次产生云端费用。
4. Shape L2 可以识别全局尺度差，但不能证明绝对场强、单位或能量守恒正确。
5. 复杂曲面场景尚未进行系统网格收敛，因此当前差异中可能混合了不同离散方式的有限分辨率误差。

## 15. 最终结论

当前 `feature_coverage` 在**场景覆盖、真实对象门禁、reference 完整性和端到端可执行性**方面已经完整，不建议继续无目标地增加更多同类场景。下一阶段的收益主要来自修复现有红区，而不是继续扩充数量。

建议验收标准分两层：

- 覆盖层：保持 53/53 场景可构建、可导出、reference 可用、结果可提取。
- 数值层：优先把 P0 场景的量纲/归一化链路修正，再逐步提升 Shape L2、Corr、flux 和 scalar 指标；任何修复都应只重跑受影响的场景并保留增量报告。

在当前状态下，系统已经具备较宽的 Tidy3D 交叉验证框架，但还不能宣称所有已覆盖功能都达到生产级数值等价。

## 16. 数值一致性修复阶段 A 记录

本节记录原评估之后的第一阶段修复。原报告第 7--10 节保留初始测量值，便于审计；下面的数值是修复比较/量纲链路后对受影响场景的增量重跑结果。

### 16.1 已修复的比较与量纲问题

1. Tidy3D cache 中的场量按其原生 `V/um` 单位统一乘 `1e6` 转换为 SI `V/m`；不再对开启 source normalization 的 TFSF 场使用错误的特殊缩放。TFSF export contract 现在无条件进入 cache key，旧语义的 reference 会被明确拒绝。
2. TFSF 功率分母改为解析入射功率 `0.5 * |E0|^2 * A * |d_n| / eta0`。box 使用注入盒横截面积，slab 使用物理 domain 横截面积；不再从归一化语义不同的历史空场 cache 推断功率。
3. Maxwell 的高斯时域源 `env(t-delay) * cos(omega*(t-delay)+phase)` 现在精确映射到 Tidy3D：`phase_td = omega*delay - phase - pi/2` 且 `remove_dc_component=False`。独立波形采样验证得到相关系数 1、相对 L2 `3.4e-15`；旧映射的相应结果约为 0.893 和 0.457。
4. time-monitor cache 保存物理 `t`；比较器只在两个求解器的公共物理时间窗内插值、逐轨迹归一化，并报告零延迟 L2 和受限互相关延迟诊断。旧 cache 缺少 `t` 时可以按场景步长重建时间轴，但不会再把两个不同长度的轨迹分别拉伸到 `[0, 1]`。
5. 电场视觉检查成为阶段验收的一部分。`field_comparison.png` 在每个 `Ex/Ey/Ez` 切片标题中显示绝对峰值；`complex_field_diagnostic.png` 的相位只在 Tidy3D 参考场的有效激励区域显示，避免近零分量经独立色标放大后形成误导。

### 16.2 修复后的增量指标

| Scenario | Field L2 | Shape L2 | Corr | Flux err |
|---|---:|---:|---:|---:|
| `tfsf_vacuum` | 0.077349 | 0.015399 | 0.9999 | `5.34e-13` |
| `tfsf_dielectric_sphere` | 0.47143 | 0.47129 | 0.8820 | 0.047041 |
| `bloch_oblique_te` | 0.19420 | 0.19059 | 0.9817 | 0.18510 |
| `rcs_pec_sphere` | 0.22131 | 0.18223 | 0.9833 | - |
| `rcs_dielectric_box` | 0.18206 | 0.17457 | 0.9846 | - |

RCS 标量不再全部饱和为 error=1：PEC sphere 的 forward/back/E/H broadside error 分别为 0.27685、0.036084、0.52366、0.14725；dielectric box 分别为 0.10143、0.28278、0.60023、0.091458。结果仍未全部达标，但已经证明原 P0 现象主要由场单位错误触发，而不是 near-to-far 后处理整体失效。

### 16.3 电场切片视觉结论

- `tfsf_vacuum`：主 `Ex` 的幅值、实部、相位和中心线高度一致；Tidy3D 有约 10% 的小幅纹波。`Ey/Ez` 的外观差异来自 `1e-8`--`1e-7` 量级的近零交叉极化分量，不能按各自自动色标判为主场失败。
- `tfsf_dielectric_sphere`：整体散射拓扑一致，差异集中在球面材料界面、TFSF box 表面和穿球中心线附近。Maxwell TFSF 与同网格 Maxwell soft source 的场型差仅约 0.038，因此注入器不是主要矛盾；需要在阶段 C 做曲面/subpixel 网格收敛。
- `bloch_oblique_te`：主传播方向和大尺度波前一致，但存在弥散的约 0.2 残差，并在介质界面局部增强。这与离散源相位和 boundary wrap 相位契约不统一的代码级证据一致。
- 两个 RCS 场景：有物理意义的电场切片与中心线形状相近，残差主要位于散射体表面和 TFSF 边界。当前证据不支持把 near-to-far 算法标记为根本性不兼容；先保留为界面离散/闭合面输入精度问题。

### 16.4 充分验证后标记的算法差异

- `DEFERRED(bloch-discrete-wavevector)`：TFSF provider 当前用离散数值波数同时缩放完整传播方向，而 Bloch boundary wrap 使用显式连续横向波数。实测 x/y 横向相位不闭合约 0.040/0.070 rad。正确修复需要固定物理 `kx, ky`、只从 Yee dispersion 解 `kz`，并统一 source delay、E/H 面和 boundary wrap；这不是比较器缩放补丁，留作独立算法阶段。
- `DEFERRED(mode-eigensolver/injection)`：原 `mode_source_higher_order` 使用方形芯层，存在模式简并；Tidy3D 按请求极化族排序，而 Maxwell 的顺序包含两个简并偏振成员，且原 `y=0` 的 `Ez` 切片恰好落在参考模式节点。即使强制枚举 Maxwell 候选模式，最佳横向重叠仍只有约 0.32，说明除场景病态外仍有本征模/注入差异。阶段 B 将改用非简并矩形波导、横向全矢量 overlap 和 2--3 级网格收敛；若 `n_eff` 相对误差仍大于 1% 或 overlap/purity 小于 0.95，则保持该标记并另案修复。

### 16.5 Time-monitor 最终验证

`time_monitor_vacuum` 的旧 reference 使用错误高斯相位且没有物理时间坐标，已被新的 source-time cache contract 正确判为 stale。外部上传在代理环境中被租户安全策略阻断，因此由用户在本地显式执行 reference 重算；随后用 `WITWIN_BENCHMARK_NO_CLOUD=1` 验证新 cache 命中，保证评分和重绘没有二次云提交。

修复后的结果：

| 项目 | 结果 |
|---|---:|
| frequency-domain `field/Ex` L2 / Shape L2 / Corr | 0.13223 / 0.098152 / 0.9952 |
| `field_time_Ex` waveform L2 | 0.18827 |
| `field_time_Ex` 最佳延迟诊断 | -70.175 ps；L2 0.11726 |
| `flux_time` waveform L2 | 0.080307 |
| `flux_time` 最佳延迟诊断 | 0 ps；L2 0.080307 |

`time_trace_comparison.png` 的视觉检查显示：约 7--11 ns 的主电场脉冲在包络、载波相位和峰值时刻上重合，FluxTimeMonitor 的各功率峰几乎逐点重合；field waveform 的主要剩余误差来自 Maxwell 在约 12--14.5 ns 的低幅尾部振铃，而不是源时间原点或高斯相位错误。真空本地解析传播验证进一步检查了峰值时刻 `source_delay + distance/c`：field 和 flux 峰均落在预期值的 1.5 个 monitor sample step 内。

阶段 A 最终回归包括：Tidy3D adapter / benchmark validation `231 passed`，受影响 runtime 测试 `108 passed, 1 skipped`，以及真实 Tidy3D `GaussianPulse.amp_time` 与 Maxwell `GaussianPulse.evaluate` 的 257 点逐点一致性测试。

## 17. Numerical-consistency repair stage B: modal system

Stage B separates a confirmed public-contract bug from a deeper eigensolver algorithm difference. The contract fix is retained; the algorithm difference is documented rather than hidden by changing the requested mode number.

### 17.1 Confirmed contract and validation fixes

1. `ModeSource`, `ModeMonitor`, and `ModePort` now define `mode_index` within the requested tangential-electric polarization family. Full-vector candidates use `sum(|E_preferred|^2) / sum(|E_tangential|^2) >= 0.5`, remain ordered by descending propagation constant within that family, and fail explicitly when the family does not contain the requested order. They no longer silently fall back to an orthogonal mode.
2. Tidy3D mode export retains the matching `TE_fraction` / `TM_fraction` sort contract and expands its candidate window from `2 * (mode_index + 1)` to `2 * (mode_index + 4)`. The modal export cache contract was incremented so old cloud references cannot be reused under the new ordering.
3. Tidy3D field monitors now request `colocate=True` explicitly. Maxwell benchmark extraction retains the raw Yee-component fields for existing scalar diagnostics and additionally exposes the solver's co-located multi-component plane payload.
4. Modal field comparison now aligns all requested E components to one physical transverse grid and computes a trapezoid-area-weighted complex vector overlap with one global phase. It also reports raw vector L2, best-one-complex-scale shape L2, Linf, and electric-energy ratio. Per-component phase fitting is forbidden because it would conceal polarization and relative-component-phase errors.
5. `vector_field_comparison.png` plots the full transverse electric magnitude and signed real components for Maxwell, Tidy3D, and the globally aligned residual. The non-degenerate modal coverage scene uses a finite x-normal output plane, so the comparison is a real cross section instead of the previous longitudinal `y=0` slice that coincided with a higher-order-mode node.

Regression coverage includes requested-family ordering, orthogonal-polarization rejection, relative-component-phase sensitivity, zero-field invalidation, co-located payload extraction, physical-coordinate alignment, adapter candidate counts, and explicit Tidy3D colocation. Targeted results before the external rerun: mode solver `10 passed`, Tidy3D adapter `133 passed`, benchmark system `84 passed`; modified-source Ruff checks pass.

### 17.2 Three-grid eigenspectrum evidence

The replacement baseline uses a rectangular `0.30 m x 0.15 m` core, a `0.525 m x 0.375 m` source/monitor aperture, `2 GHz`, `polarization="Ez"`, and public `mode_index=1`. The geometry is exactly representable on all three study grids.

| Grid spacing (m) | Second physical Ez-mode group n_eff | Ez purity range | Gap to nearest different physical group |
|---:|---:|---:|---:|
| 0.0375 | 1.912182 | 0.990--1.000 | 0.02121 |
| 0.0250 | 1.900120 | 0.996--1.000 | 0.03625 |
| 0.0125 | 1.891643 | 0.997--0.999 | 0.03989 |

The physical group changes by 0.448% from the medium to fine grid and 1.086% from coarse to fine. It is non-degenerate with the neighboring physical mode and has one signed `Ez` node along y and none along z. This establishes a stable target independent of the original square-guide polarization degeneracy.

### 17.3 `DEFERRED(mode-eigensolver-checkerboard-duplicates)`

The current full-vector transverse operator composes centered first derivatives. On the transverse lattice this decouples odd/even sublattices and produces approximately four checkerboard copies of each continuous physical mode. At `dx=0.025 m`, raw Ez-family candidates 0--3 all have zero center-line nodes, while candidates 4--7 all have one y node; the four eigenvalues inside each physical group collapse toward one value as the grid is refined.

Consequently, public `mode_index=1` currently selects another zero-node fundamental copy. The local Stage-B Maxwell run returns `n_eff=1.9408339`, whereas the independently grouped second physical Ez mode is near `1.90012` on the same grid. Changing the public request to `mode_index=4` would only disguise the defect and would ask Tidy3D for a different physical order, so that workaround is explicitly rejected.

This item is sufficiently verified as a fundamental mode-eigensolver discretization difference. It remains deferred for a dedicated operator/de-duplication redesign. The new Tidy3D reference, full-vector overlap, `n_eff`, cross-section plot, and signed propagation slice remain the acceptance baseline for that future repair.

### 17.4 External reference and electric-field visual verdict

The user regenerated the Tidy3D reference under modal export contract version 2. A subsequent `WITWIN_BENCHMARK_NO_CLOUD=1` run hit cache key `ec917b965d219b04a39913bb5071c2f1f77a7d8cf0e12ac9697835031b8db939`, proving that scoring and plotting used the new reference without another cloud submission.

| Quantity | Maxwell | Tidy3D | Comparison |
|---|---:|---:|---:|
| `n_eff` | 1.940834 | 1.889471 | relative error 0.026464 |
| electric-vector energy | - | - | Maxwell/Tidy3D ratio 0.2674 |
| full-vector field | - | - | raw L2 0.97992; shape L2 0.97812; overlap 0.2081 |
| transmitted flux | - | - | relative error 0.93845 |

The transverse electric-field image is decisive rather than merely illustrative. Tidy3D launches the requested signed second-order `Ez` mode: two transverse lobes with a sign reversal at the central node. Maxwell launches a single central lobe with no transverse node and visible staggered/checkerboard leakage in the weaker components. On the longitudinal `y=0` slice, Tidy3D is consequently near zero because the plane lies on the physical mode node, while Maxwell retains a strong propagating field. One global complex alignment cannot transform either topology into the other, which is consistent with the low vector overlap and rules out a simple unit, phase, or normalization explanation.

Stage B therefore closes with the public polarization-family contract fixed and regression-tested, while the eigensolver checkerboard duplication remains explicitly deferred. The generated `vector_field_comparison.png` is the authoritative visual baseline for the future operator redesign; the old scalar longitudinal slice is not accepted on its own.

## 18. 数值一致性修复阶段 C：几何、设计区域与功率基准

阶段 C 只修改经过独立控制实验确认的局部错误。模式本征算子、Bloch 离散波矢以及 full-tensor 界面算子不通过经验系数或场景特判掩盖，继续保留为独立算法任务。

### 18.1 Cone 导出语义修复

共享 `Cone` primitive 的 `position` 是顶点，几何沿正轴从顶点延伸一个 `height`。旧 Tidy3D 适配器却把顶点直接作为 tapered `Cylinder.center`，使用正 `sidewall_angle` 和默认 middle reference plane。以 `cone_scatter` 为例，Maxwell 几何范围是 `z=[-0.12, 0.12]`、最大半径 `0.15 m`，旧导出则约为 `z=[-0.24, 0]`、最大半径 `0.225 m`；在 `dx=0.025 m` 的硬 mask 对照中 IoU 仅约 0.025。因此旧 Field L2 0.59606、Shape L2 0.56047 和场图中的刚性轴向错位不能作为求解器离散误差证据。

修复后，未旋转 cone 沿所选轴把 Tidy3D center 平移 `height/2`，使用负锥壁角和 `reference_plane="top"`；任意旋转 cone 走 triangle-mesh 精确回退。x/y/z 三轴 mock 测试与真实 Tidy3D `bounds`/`inside` 测试均通过。当前 benchmark 网格上 Maxwell/Tidy3D 导出 mask 均为 337 个节点，XOR 为 0、IoU 为 1.0；三档静态对照的 IoU 为 1.000/0.994/0.998，剩余仅是落在边界上的浮点判定。

### 18.2 诊断图的假阳性修复

旧 `material_source.png` 中所谓 Tidy3D material 实际来自 `prepare_tidy3d_benchmark_scene(scene)` 返回的 `scene.clone()`，因此差值按构造接近零，甚至会把上述错误 cone 导出显示为“材料完全一致”。新诊断图不再作这个声明：它显示 Maxwell 编译介电率、Maxwell 公共几何硬 envelope、调用真实适配器后在同一 Maxwell 网格采样的 Tidy3D geometry envelope、两者 XOR/fraction，以及 Maxwell 源 stencil。shifted-cone 回归证明该图会对刚性错位产生非零 XOR；完整 106 个场景的本地导出采样均成功。

### 18.3 MaterialRegion 亚像素编译修复

旧 MaterialRegion 在所有 subpixel 样本平均完成后才用硬 node slice 覆盖材料，并从三轴平均的 `eps_r/mu_r` 计算一个共同增量再加回 x/y/z。这同时绕过 soft occupancy、Kottke polarized averaging，并在各向异性背景上保留错误的轴间偏差。

现在 density 在原生设计网格上只执行一次 normalize/filter/projection，全程留在 scene device；每个 subpixel 坐标通过 PyTorch `grid_sample` 采样 density texture，并复用普通 Structure 的 geometry occupancy、interface normal、arithmetic/polarized blend。每一 Yee 轴从其实际当前背景混合到 density 定义的标量 `eps/mu`，覆盖区域中的 conductivity、off-diagonal、dispersion、nonlinear 和 modulation channel 也按等价普通标量 Structure 的语义被置换。多样本聚合仍保存 `eps_r_base/mu_r_base`、最终减 base 的 design tensor 和名义 hard design mask。

非网格对齐 Box 在 `samples=1/3`、arithmetic/polarized、对角各向异性底层材料和非平凡 `eps/mu` bounds 下，与等价普通 Structure 的全部六个 `eps/mu` 分量在 `1e-6` 内一致；density 梯度保持 finite 且非零。使用原 Tidy3D reference 的本地增量结果为：

| Scenario | Field L2 | Shape L2 | Corr | Flux err |
|---|---:|---:|---:|---:|
| `material_region_slab`（修复前） | 0.22325 | 0.22315 | 0.9748 | 0.137 |
| `material_region_slab`（修复后） | 0.15706 | 0.15701 | 0.9876 | 0.11751 |

新 geometry XOR 三个正交切片均为 0。复数 `Ex` 图显示传播拓扑和中心线实部已高度重合，主要残差收缩到 slab 两格界面带、源/PML 公共底部条带以及弱的外部纹波；它已不再表现为不同 material extent，但普通 slab 的界面/共点离散差异仍未达到 Shape L2 0.10 的门槛。

### 18.4 强驱动场景的 incident-power 分母

Tidy3D `FluxData` 即使在 field monitor 使用 `normalize_index` 时仍保存物理功率。旧 incident-cache signature 包含 PlaneWave source-time amplitude，导致强 Kerr/TPA 场景无法匹配 amplitude-one vacuum reference，随后错误地用散射/透射 reference flux 自身作分母。新 signature 只忽略单一 PlaneWave 的幅值、绝对相位和单位横向偏振方向，仍严格保留 frequency/fwidth/delay、direction/injection、domain/grid/boundary/symmetry 和 Courant；匹配后的空场物理功率始终乘 `|amplitude|^2`。

为了不把 PML 横向边界的空场功率用于 periodic 非线性场景，新增 `planewave_periodic_vacuum` reference。它与 `kerr_slab`、`kerr_slab_strong`、`tpa_slab` 和 `dispersive_kerr_slab` 的发射/网格/边界物理匹配；PML `planewave_vacuum` 继续只服务相同 PML signature。该改动修正指标定义，不保证数值一定变小。

### 18.5 已验证后暂缓的算法差异

- `DEFERRED(full-tensor-interface-collocation)`：adapter 的 3x3 tensor 行列映射正确，whole-domain tensor 和本征偏振控制通过，而有限 slab 的 Ex/Ey/Ez 系统残差集中在界面；当前 compiler 明确没有 off-diagonal tensor 的 polarized interface averaging，runtime 又在 staggered Yee 场上单独应用 off-diagonal correction。正确修复需要完整 tensor-interface averaging、field colocation 和匹配 adjoint，不能用源旋转或系数拟合代替。
- `DEFERRED(mesh-interface-subpixel-convergence)`：cylinder、pyramid/explicit-mesh、torus 和 hollow-box 的公共硬导出 mask 已验证一致；pyramid 与由其 `to_mesh()` 生成的 explicit mesh 编译差仅 `1e-6` 量级。现有差异属于共享 triangle-mesh/curved-interface occupancy 与 Tidy3D subpixel 算子的收敛问题。hollow-box 壁厚 `0.05 m` 在当前网格只有两格，必须先细化到 `0.0125 m` 或保证 3--4 cells/壁厚，不能据当前 Shape L2 0.510 宣称独立 geometry bug。
- `DEFERRED(time-modulation-validation)`：modulation 的 amplitude/phase adapter 映射、`a*cos(phi)/a*sin(phi)` compiler 通道和 conservative D update 相互一致，未发现安全的局部错误。后续必须按 carrier/lower/upper sideband 分频绘图，并做 depth=0、弱调制、运行时间与网格收敛后再判断 finite-window 还是更新算法差异。

### 18.6 External references, incremental metrics, and visual verdict

The user regenerated `cone_scatter` and `planewave_periodic_vacuum` with the current cache contracts. The following run set `WITWIN_BENCHMARK_NO_CLOUD=1`; both references were cache hits, so scoring and plotting did not submit another cloud task.

| Scenario | Field L2 | Shape L2 | Corr | Flux err |
|---|---:|---:|---:|---:|
| `cone_scatter` (old invalid export) | 0.59606 | 0.56047 | 0.8282 | 0.069713 |
| `cone_scatter` (correct export) | 0.17949 | 0.17949 | 0.9838 | 0.012849 |
| `planewave_periodic_vacuum` | 0.0023268 | 0.0023264 | 1.0000 | 0.0019993 |
| `kerr_slab` | 0.054233 | 0.054176 | 0.9985 | 0.11318 |
| `kerr_slab_strong` | 0.14146 | 0.13845 | 0.9904 | 0.18946 |
| `tpa_slab` | 0.087088 | 0.085988 | 0.9963 | 0.084146 |
| `dispersive_kerr_slab` | 0.40085 | 0.37835 | 0.9257 | 0.26163 |

The incident-reference correction is most visible in the flux metrics. `kerr_slab` changed from the invalid self-reference value 3.7796 to 0.11318, and `kerr_slab_strong` changed from 1.6381 to 0.18946. `tpa_slab` changed only from 0.085797 to 0.084146 because its previous accidental denominator happened to be close. `dispersive_kerr_slab` changed from 0.34569 to 0.26163. These are semantic corrections to the denominator; they are not solver-output fitting.

The electric-field images support the same conclusions:

- `cone_scatter`: the scattered-wave fronts, shadow, and peak location now agree. The old rigid axial displacement is absent. The remaining residual is concentrated at the cone interface, the source/PML strip, and the local scattering amplitude. The geometry diagnostic reports zero XOR on all three orthogonal slices.
- `planewave_periodic_vacuum`: all three principal `Ex` slices are visually coincident, while `Ey/Ez` remain zero. This validates the periodic incident-power reference independently of the nonlinear media.
- `kerr_slab`, `kerr_slab_strong`, `tpa_slab`, and `dispersive_kerr_slab`: the dominant `Ex` propagation topology agrees in the longitudinal slices. Strong Kerr and dispersive Kerr retain visible slab-interior amplitude/phase residuals, consistent with their 0.138 and 0.378 Shape L2 values. Maxwell also has weak transverse-component leakage that appears prominent only when those near-zero components receive their own color scale. Therefore only the weak Kerr and TPA dominant fields are considered close; the strong and dispersive cases remain numerical-improvement targets.

### 18.7 Stage-C regression closure

Targeted regression results are: material compiler `32 passed`, Tidy3D adapter `138 passed`, and benchmark validation `92 passed`. Ruff passes on the modified production modules and the new tests (the adapter test retains only its explicitly listed legacy ignores), and `git diff --check` is clean. The external-cache run above is the final numerical and visual acceptance pass for this stage.

## 19. Numerical-consistency repair stage D: boundaries, grids, and spectral diagnostics

Stage D fixes two independently reproduced validation defects and makes the visual
acceptance criteria frequency-aware. It does not fit solver outputs or weaken field
metrics. Confirmed constitutive/source algorithm differences remain explicit deferred
items.

### 19.1 Exact subpixel interfaces and periodic endpoint planes

Multi-sample material compilation previously counted a sample exactly on an analytic
interface as fully inside. Odd sample counts therefore biased a grid-aligned interface
toward the structure material. For a vacuum-to-epsilon-3 interface, sample counts
2/3/4/5 produced tangential epsilon values near
`2.000/2.333/2.014/2.201` instead of a sample-count-independent value of 2.
Exact signed-distance boundary samples now receive half occupancy.

The old periodic repair added translated geometry by a domain span. A structure that
already covered one or more periods consequently double-counted its image throughout
the volume, while removing images entirely would lose a shorter structure that really
crosses the seam. The compiler now takes a differentiable maximum union of the base and
periodic images. It then composes the duplicate endpoint planes, using adjacent interior
planes (or the cell midpoint for a legal two-node periodic axis) to distinguish a real
wrap continuation from an orthogonal partial interface. This has no per-sample CUDA
`.item()` reachability synchronization. Static world-space geometry bounds restrict the
image set to faces the geometry can actually reach; trainable or unknown-bound geometry
retains the exhaustive differentiable union. Exact half values use a straight-through
value correction, so trainable geometry gradients are retained.

Regression coverage uses an over-period slab and checks the center, both periodic
endpoints, and periodic corners for sample counts 2/3/4/5. Separate tests verify that a
short box crossing one seam fills the opposite-side interior, a full-period structure
works on a two-node periodic axis, and the exact-interface half value retains a finite
nonzero size gradient.

The field and flux results confirm that this is a material-occupancy repair rather than
a boundary-kernel change:

| Scenario | Old Shape L2 | New Field L2 | New Shape L2 | New Corr | New flux err |
|---|---:|---:|---:|---:|---:|
| `periodic_slab` | 0.3095 | 0.0023315 | 0.0023314 | 1.0000 | 0.0059300 |
| `mixed_faces` | 0.3423 | 0.0084299 | 0.0034108 | 1.0000 | 1.4444 |

All three principal electric-field cuts for `periodic_slab` are visually coincident.
The `mixed_faces` Maxwell/Tidy3D `Ey` cuts have the same uniform one-dimensional
topology and their phase-aligned center lines are nearly indistinguishable. A control
that replaces the y-PEC faces by y-periodic faces gives Shape L2 below `2e-6`, proving
that the mixed periodic/PEC/PML update kernel is not the source of the remaining flux
outlier.

### 19.2 Nominal infinite slabs must continue through transverse absorbers

`Domain.bounds` is the physical domain, while PML cells are appended outside it. The
shared plane-wave benchmark helper previously ended its nominally infinite slab at the
physical x/y bounds. This created a finite plate edge exactly at the transverse PML
entrance. On graded grids it generated strong transverse diffraction and checkerboard
patterns unrelated to the intended one-dimensional grid/material comparison.

The helper now extends a slab through the computational PML only on transverse axes
that use an external absorber. Periodic and Bloch axes retain exactly one physical
period. A regression checks both the structure bounds and the compiled material slice
over the prepared PML domain.

The user force-regenerated all eleven affected Tidy3D references. A subsequent run with
`WITWIN_BENCHMARK_NO_CLOUD=1` validated every new cache key and used cache hits only:

| Scenario | Field L2 | Shape L2 | Corr | Flux err |
|---|---:|---:|---:|---:|
| `pec_box` | 0.025124 | 0.023072 | 0.9997 | 0.033387 |
| `full_tensor_slab` | 0.63632 | 0.59138 | 0.8064 | 0.62443 |
| `sigma_e_slab` | 0.058696 | 0.058674 | 0.9983 | 0.047985 |
| `custom_pole_uniform_slab` | 0.099650 | 0.099544 | 0.9950 | 0.10134 |
| `perturbation_uniform_slab` | 0.040047 | 0.039989 | 0.9992 | 0.033508 |
| `sellmeier_slab` | 0.14172 | 0.14065 | 0.9901 | 0.034228 |
| `custom_grid_slab` | 0.015353 | 0.014813 | 0.9999 | 0.084003 |
| `autogrid_slab` | 0.035718 | 0.035597 | 0.9994 | 0.016964 |
| `nonuniform_custom_grid` | 0.069794 | 0.069718 | 0.9976 | 0.012579 |
| `anisotropic_uniform_grid` | 0.008008 | 0.007654 | 1.0000 | 0.029319 |
| `autogrid_override_refinement` | 0.045899 | 0.043599 | 0.9990 | 0.008327 |

The five grid-scene electric-field images now show a purely one-dimensional `Ex`
propagation pattern: x/y-normal cuts vary only with z, z-normal cuts are transversely
uniform, and `Ey/Ez` remain at numerical zero. The old domain-wide checkerboard and PML
entrance diffraction are absent. Phase-aligned real fields and unwrapped center-line
phases overlap closely; residuals are smooth z bands plus the shared source-plane band.
In particular, `nonuniform_custom_grid` improved from Field/Shape L2
`0.13829/0.13773` to `0.069794/0.069718`, and its correlation increased from 0.9905 to
0.9976. `anisotropic_uniform_grid` improved from 0.09555 to 0.00801 Field L2, while
`autogrid_override_refinement` improved from 0.15905 to 0.04590. There is no remaining
evidence for a fundamental nonuniform-Yee stencil error in these scenes.

The full-tensor image gives the opposite verdict. Maxwell and Tidy3D retain the same
eigenpolarization ratio (`Ey/Ex` approximately 0.351/0.350), longitudinal phase trend,
and one-dimensional propagation topology, but their finite-slab standing-wave amplitude
and interface impedance differ systematically. Maxwell also produces an unexpected
`Ez` peak near 0.184 where Tidy3D remains near 0.003. Together with
the whole-domain tensor control and correct adapter matrix mapping from Stage C, this
strengthens `DEFERRED(full-tensor-interface-collocation)`: the unresolved work is the
full-tensor Yee-interface/colocation operator and its adjoint, not scene export.

### 19.3 Frequency-aware electric-field diagnostics

Multi-frequency summaries already selected the worst field metric, but the old main
image always plotted frequency index zero. Worse, modulation scenes independently fit
a global phase at every sideband, which erased relative carrier/sideband phase errors.
The runner now:

1. selects the worst-L2 frequency for the main field image;
2. uses the nearest extracted resonance for resonance-observable diagnostics;
3. derives one global phase anchor from the carrier and applies it unchanged to all
   modulation frequencies; and
4. writes a three-row spectral diagnostic containing Maxwell/reference magnitude,
   signed real field, complex residual, and per-frequency metrics.

The shared carrier anchor produces the following authoritative modulation results:

| Scenario / frequency | Field L2 | Shape L2 | Corr |
|---|---:|---:|---:|
| `modulated_slab` / 1.8 GHz | 0.30194 | 0.27054 | 0.9627 |
| `modulated_slab` / 2.0 GHz carrier | 0.18542 | 0.10943 | 0.9940 |
| `modulated_slab` / 2.2 GHz | 0.45534 | 0.14729 | 0.9891 |
| `modulated_slab_phase` / 1.65 GHz | 0.18104 | 0.16766 | 0.9858 |
| `modulated_slab_phase` / 2.0 GHz carrier | 0.19590 | 0.13116 | 0.9914 |
| `modulated_slab_phase` / 2.35 GHz | 0.38005 | 0.33198 | 0.9433 |

All six electric-field images preserve the same one-dimensional sideband topology, but
the upper sidebands retain systematic amplitude and relative-phase residuals. This is
visible directly under one carrier phase anchor and is no longer hidden by independent
phase fitting. No DFT-window, Hann normalization, adapter phase-sign, or safe local
runtime defect was found, so `DEFERRED(time-modulation-validation)` remains in force
for depth, duration, and grid convergence followed by a constitutive-update audit.

`pmc_cavity` is no longer classified from the worst off-resonance noise slice. The
Maxwell/Tidy3D extracted resonances are 210.9960 and 210.4843 MHz, a relative difference
of 0.2425%; both are within 0.71% of the analytic 211.9853 MHz `Hz(110)` resonance. The
new diagnostic selects the 210 MHz sample. Its dominant electric-field lobes and Hz
mode topology agree visually (Shape L2 0.0656 at that sample), so the old aggregate
Shape L2 0.7118 over a broad off-resonance sweep is retained as a worst-frequency
number but is not evidence of a PMC boundary defect.

### 19.4 Explicitly deferred source/monitor semantics

- `DEFERRED(mixed-soft-source-upstream-net-flux)`: the mixed-face field is resolved and
  the transmitted powers differ by about 0.8%, but the upstream total-field monitor
  gives opposite signed net values near incident/reflected cancellation. It is not an
  isolated reflected-power observable. Separating incident and scattered flux, and
  auditing the soft-source E/H power balance, is required; a sign flip or excluding the
  monitor from the maximum would only conceal the discrepancy.
- `DEFERRED(gridded-current-source-discretization)`: the magnetic-current image has the
  same antisymmetric radiation-ring topology and source center in both solvers, while
  electric gridded-current controls are also discrepant. Half-cell translation does not
  resolve it, and strict clipping/control-volume overlap candidates worsen Shape L2
  from 0.369 to 0.784/0.606. A smooth zero-edge dataset, multi-grid convergence, and an
  electromagnetic-duality test are needed before changing injection semantics.
- `DEFERRED(custom-grid-flux-sampling)`: `custom_grid_slab` has excellent field agreement
  (Shape L2 0.0148, correlation 0.9999) but flux error 0.0840. This is isolated from the
  repaired transverse field topology and should be checked by moving the flux plane
  away from a grid transition and comparing the colocation/quadrature aperture.

### 19.5 Stage-D regression closure

Material-compiler plus benchmark-validation coverage passes `134` tests (`37 + 97`).
The complete boundary suite passes `61` tests, and custom/uniform electric and magnetic
current source definitions pass `13` tests. Ruff passes on every modified production
and test module, and `git diff --check` is clean. The refreshed-reference run above is
the final numerical baseline; the
orthogonal electric-field slices, complex diagnostics, resonance-selected cavity image,
and carrier-anchored spectral images are the visual acceptance artifacts for this stage.
