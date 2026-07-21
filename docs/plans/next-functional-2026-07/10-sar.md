# SAR 与人体暴露分析开发计划

> 状态：in-progress (phases delivered 2026-07-19)  
> Delivered 2026-07-19 (Wave D selective start, owner-authorized): Phases 0–3
> (spec/mass-density contract, point SAR, 1 g/10 g mass averaging, normalization +
> coherent/incoherent multi-source combination) plus a Phase-4 slice (`soft_peak`
> surrogate + finite-difference gradient gates). Evidence is E1 (analytic /
> golden / brute-force parity / grid-convergence gates, no external reference
> solver cross-check); NOT `completed`. See
> `docs/assessments/b10-sar-acceptance-2026-07-19.md` and `tests/sar/`.  
> **Round-H revision (2026-07-21, master `6f3b0c8`; merge `8ebaec0`).** Delivered:
> `IncidentPowerDensityMonitor` + `Result.incident_power_density` (was fail-closed) with
> plane-wave `|S|=|E|²/(2·eta)` analytic gates; the redistributable canonical phantom family
> (`uniform_lossy_cube`, `layered_slab`, `one_gram_cube`, `antenna_near_phantom`) + SAR RESULTS
> rows via `python -m benchmark sar` (`layered_slab` wave-level surface/volume conservation
> closure 16.7% at dx=4 mm, converging). Cleanup honestly reclassified the `uniform_lossy_cube`
> volume/channel closure a tautology and recorded the blocked `antenna_near_phantom` wave-level
> target as target-only. Evidence E1–E2 (no external reference). **Still open:** IEEE/IEC
> *certified* phantom profiles + external cross-check (deferred), `input_power` normalization
> (fail-closed), `antenna_near_phantom` conductive-media port blocker, VOP + multi-GPU (P5).
> Phase-status bookkeeping / any `completed` mark is the supervisor's job (audit §4
> non-author-review bar), not set here. See
> `docs/assessments/h3-sar-phantom-acceptance-2026-07-21.md`.  
> 路线定位：后续垂直能力，不作为当前 RF 闭环的近期交付承诺  
> 日期：2026-07-14  
> 目标证据：E3 production  
> 路线优先级：P2（RF 基础闭环、功率密度 monitor 和端口 accepted power 之后）  
> 主要依赖：`01-rf-engineering-workflow.md` Phase 4 的共享 `PowerLossData`、组织材料数据、可审计的体素积分；Bioheat 可复用本文 Phase 1，但不依赖 SAR 质量平均/VOP
> Owner modules：`media.py`、`monitors.py`、拟新增 `postprocess/sar.py`、`result.py`  
> 最近架构决策：2026-07-14，标准 peak 1 g/10 g 与可微 soft-peak 使用不同结果字段，不以点 SAR 冒充法规平均值

## 1. 背景与当前能力

两份功能缺口分析都指出，Maxwell 已有 GPU-first Yee-grid FDTD、频域场 monitor、通量/闭合面 monitor、近远场和部分功率后处理，但人体暴露工作流仍是空白。当前 `Material` 有介电率、导电率、色散和各向异性等 EM 参数，`Scene` 能承载结构、材料、源、monitor 与 metadata，`Result` 保留 torch tensor；尚无质量密度 `rho`、组织身份、SAR、1 g/10 g 平均、法规归一化或人体体素映射。

SAR 不能实现成 `sigma * abs(E) ** 2 / (2 * rho)` 的孤立便利函数。Yee 分量不共点、组织边界存在部分体素、质量平均窗口必须连续且不能跨空气“凑质量”，结果还依赖 RMS/峰值约定与输入功率归一化。该项目的产品目标是生成可复现、带单位和算法 provenance 的暴露结果，而不是仅提供一张热力图。

## 2. 目标与非目标

### 2.1 目标

- 为有损介质增加质量密度和组织语义，不破坏现有 EM `Material` 合同。
- 生成点 SAR、局部质量平均 SAR、峰值位置、组织/区域统计和 incident/absorbed power density。
- 支持 1 g、10 g 及用户指定质量，明确 IEEE/IEC 兼容的 averaging profile。
- 支持按源幅度、端口 accepted power、总输入功率或目标发射功率缩放。
- 保持场、SAR、平均核与缩放运算为 torch tensor；可微参数不被隐式 detach。
- 为多源组合、VOP 和人体体素数据留出稳定扩展点。

### 2.2 非目标

- 本项目不求解温升、灌注或组织损伤；这些属于 `11-bioheat.md`。
- 不随库分发受许可限制的人体模型或专有组织数据库；只定义导入/映射合同和公开 benchmark fixture。
- 第一版不宣称 HAC、MRI 全工作流或任何地区的自动合规认证。
- 不用 CPU-only 医学图像路径替代 GPU 计算；DICOM/voxel 文件解析可在 CPU 发生，但编译后的数值数据必须驻留目标设备。

## 3. 用户工作流与 Public API 草案

保持单一入口：场景仍由 `Scene` 声明，FDTD 仍由 `Simulation.fdtd(...)` 执行，SAR 从标准 `Result` 派生。

```python
import witwin.maxwell as mw

skin = mw.Material(
    permittivity=41.4,
    conductivity=0.87,
    mass_density=1100.0,       # kg / m^3
    metadata={"tissue": "skin"},
)

scene = mw.Scene(domain=domain, grid=grid, boundary=boundary)
scene.add_structure(mw.Structure(geometry=phantom, material=skin))
scene.add_monitor(mw.PowerLossMonitor(
    name="exposure",
    position=phantom.center,
    size=phantom.size,
    channels=("conduction", "dispersion", "total"),
    frequencies=(900e6,),
))

result = mw.Simulation.fdtd(scene=scene, frequencies=(900e6,)).run()
sar = result.sar(
    monitor="exposure",
    averaging=mw.SARAveraging(mass=(1e-3, 10e-3), profile="ieee-iec"),
    normalization=mw.PowerNormalization.accepted_power(port="feed", watts=1.0),
)
peak_1g = sar.peak(mass=1e-3)
```

建议公共对象：

- `Material.mass_density: float | torch.Tensor | None`，SI 单位 kg/m³；无密度材料不得参与 SAR。
- `TissueSpec(name, mass_density, thermal_properties=None, provenance=None)`：可选组合描述，不建立第二套材料类。
- 复用 01 的 `PowerLossMonitor/PowerLossData` 获取体积耗散 W/m3；`IncidentPowerDensityMonitor` 单独输出暴露入射 W/m2。SAR 不新增 `AbsorbedPowerDensityMonitor` 同义对象。
- `SARAveraging(mass, profile, connectivity, boundary_policy, min_tissue_fraction)`。
- `PowerNormalization.source(...)`、`.accepted_power(...)`、`.input_power(...)`。
- `SARResult(point, averaged, peaks, statistics, normalization, provenance)`，所有大数组保持设备和 dtype。

`Result.sar(...)` 是结果域操作；它不得重新隐藏运行一个求解器。若缺失所需 monitor、密度、端口功率或频率数据，必须显式失败。

## 4. 物理、离散与算法设计

### 4.1 点 SAR 与耗散功率

对采用峰值复幅的谐波场：

```text
q_e = 0.5 * E* · Re(sigma_e) · E        [W/m^3]
SAR = q_e / rho                         [W/kg]
```

若材料含磁损耗或色散极化损耗，结果需要同时给出 `electric_conduction`、`electric_dispersion`、`magnetic` 和 `total` 通道。第一阶段 SAR 的法规通道采用电组织总吸收功率，不能把 `sigma |E|²` 当成所有材料的通用损耗。时域脉冲则按明确时间窗计算瞬时/平均耗散，不与频域 RMS 公式混用。

### 4.2 Yee 共点化与部分体素

- 将六个 staggered 分量以守恒插值映射到 material cell center；禁止简单裁剪到共同 shape。
- 材料 compiler 输出 cell volume、质量密度、组织 id、occupancy fraction 和损耗系数。
- 边界 cell 的质量为 `rho * occupancy * volume`；功率也按相同 occupancy 与材料混合规则积分。
- 共形/子像素材料必须使用与 EM compiler 一致的占据率 provenance，避免场模型和 SAR 质量模型不一致。

### 4.3 质量平均

对目标质量 `m0`，平均 SAR 定义为连续组织区域内的 `sum(q_i V_i) / sum(rho_i V_i)`。实现采用 GPU 上的分层策略：规则均匀网格用 prefix-sum/卷积产生候选窗；非均匀网格和组织边界用累计质量搜索与局部加权；峰值候选再执行精确窗口构造。

`profile` 固定算法细节：窗口形状、中心规则、空气排除、组织连通性、边界补偿和不足质量处理。结果记录实际质量、体素集合摘要和 profile 版本，禁止默默将不足 1 g/10 g 的区域补零。

### 4.4 多源、功率密度与 VOP

- 同频相干源必须用复场叠加后求 SAR；非相干源按功率加和，不能混用。
- incident power density 使用 Poynting vector 和标准指定空间平均；absorbed power density 来自材料损耗。
- VOP 放在后续 phase：输入是通道 Q 矩阵/压缩误差界，输出必须保留 overestimation bound。

## 5. 数据模型与编译边界

`Scene` 不新增 SAR 专用场景类型。材料和 monitor 经 `compile_material_tensors(...)` / `compile_*` 进入统一 prepared scene：

```text
Material.mass_density / tissue id
        -> compile material tensors
        -> rho_cell, tissue_id, occupancy, loss_channels, cell_volume
FDTD fields + port/source power
        -> absorbed-power monitor
        -> SAR reducer
        -> SARResult
```

建议 payload 至少包含 `values`、`units`、`coordinates`、`frequency`、`field_convention`、`material_mapping_hash`、`normalization`、`averaging_profile/version`、`grid_hash`。保存/加载沿用 `Result`，不能通过匿名 metadata 字典承载核心结果。

## 6. GPU-first、PyTorch 与梯度策略

- 共点化、损耗计算、质量积分、峰值搜索和区域统计在 CUDA torch/原生 kernel 上完成；最终标量展示才允许 CPU copy。
- 点 SAR、固定窗口平均、区域均值与 normalization 保持 autograd；密度、导电率、几何 occupancy 和端口功率梯度可回传。
- `argmax` 峰值位置不可微。优化 API 同时提供 `soft_peak(temperature=...)` 和固定候选区域的 differentiable maximum surrogate，并明确它不是法规 peak。
- 离散组织 id、连通域和窗口 membership 默认停止梯度；若几何优化导致 membership 改变，用户必须选择软 occupancy 近似或接受分段梯度。
- 不新增 CPU fallback；极端模型显存不足应通过 chunk/shard 和 multi-GPU 解决。

## 7. Multi-GPU contract

- 材料 cell、点 SAR 与功率密度遵循场域分片 ownership；halo 只用于共点插值和边界窗口。
- 质量平均候选窗由其中心 cell 的 owner 负责，跨 shard 所需积分量通过 halo/prefix exchange 获取，禁止复制整个人体体素。
- 全局 peak 使用 `(value, global_index)` 确定性 max-reduction；并列值按最小全局线性索引决胜。
- 组织/区域统计使用 GPU all-reduce；结果与单 GPU 在给定容差内一致。
- 任务级多 GPU 可并行频点、源通道或姿态；02 executor 不隐式归约独立 Result。SAR/array aggregator 必须先校验统一频率、复数相位、reference plane 和功率 normalization metadata，再显式组合相干复场；非相干源只在功率层求和。

## 8. 开发 phases、依赖与 exit gate

### Phase 0：规范冻结与数据契约（E0, experimental）

Deliverables：单位/幅值约定、材料密度通道、损耗分解公式、averaging profile schema、公开 phantom 许可清单。依赖：RF 功率约定已冻结。Exit gate：一份可执行 reference notebook 在手工小网格上逐 cell 对账；API review 通过。

### Phase 1：点 SAR 与功率密度（E1, experimental）

Deliverables：组织材料编译、共享 PowerLossData 消费、incident power density、点 SAR、区域统计、结果序列化。Exit gate：均匀有损介质解析误差达到 benchmark 阈值；耗散体积分与入射减出射功率闭合；CPU reference parity。

### Phase 2：1 g/10 g 质量平均（E2）

Deliverables：均匀/非均匀网格 averaging、部分体素、连通组织规则、peak search、provenance。Exit gate：合成质量窗逐 cell golden test、网格收敛、公开 IEEE/IEC phantom 结果在预注册容差内。

### Phase 3：归一化与多源组合（E2）

Deliverables：source/accepted/input power scaling、相干/非相干组合、参数 sweep 聚合。依赖：RF 端口 accepted power。Exit gate：缩放平方律、多源相位扫描和端口功率端到端验收全部通过。

### Phase 4：规模化与可微工作流（E3）

Deliverables：chunked/multi-GPU reducer、soft peak、组织参数梯度、性能基线。Exit gate：单/多 GPU parity；有限差分/伴随梯度检查；生产尺寸模型无隐式 host materialization。

### Phase 5：VOP 与标准工作流（E3, optional product gate）

仅在 MRI/多通道暴露产品需求明确后启动。Deliverables：Q 矩阵、VOP 压缩、误差上界和公开验证。Exit gate：所有压缩结果对训练/保留激励均满足声明的保守上界。

## 9. 验收与 benchmark 策略

- **解析**：均匀有损介质中的已知 E 场；平面波吸收；两材料分层边界。
- **离散**：staggered-to-cell interpolation、partial voxel、非均匀体积、恰好/不足目标质量的 golden tests。
- **网格收敛**：点 SAR 不作为唯一结论；1 g/10 g peak、吸收总功率和区域平均随网格至少三档收敛。
- **守恒**：`P_in - P_out - dU/dt` 与体积总耗散闭合；频域稳态采用复 Poynting 约定。
- **标准/独立参考**：采用允许再分发的 IEEE/IEC canonical phantom；与至少一个独立求解器或公开结果对照，版本固定。
- **性能**：记录 cells/s、额外显存、averaging time、peak-reduction time；SAR 后处理默认额外峰值显存预算目标不超过基础字段的 35%。
- **端到端**：有端口天线照射 phantom，从 accepted power=1 W 到 peak 1 g/10 g、组织统计、保存/加载完整复现。
- **梯度**：固定窗口 SAR 对导电率、源幅和连续密度参数的 central difference，float64 reference 用于小问题审计。

benchmark 建议加入 `benchmark/scenes/sar/`：`uniform_lossy_cube`、`layered_slab`、`partial_voxel_sphere`、`one_gram_cube`、`ten_gram_phantom`、`antenna_near_phantom`、`coherent_two_source`。`benchmark/RESULTS.md` 报告绝对值、相对误差、守恒残差、实际平均质量和运行配置。

## 10. 主要风险与缓解

- **标准解释错误**：以 versioned profile 固定算法，不以一个布尔 `standard=True` 掩盖差异。
- **材料混合导致质量/损耗不一致**：SAR compiler 消费 EM material compiler 的同一 occupancy。
- **峰值对网格敏感**：强制报告网格收敛和实际平均质量，不把 point peak 当法规结果。
- **人体数据许可**：核心测试只依赖可再分发 phantom；专有数据通过 adapter，不进入仓库。
- **梯度与法规值混淆**：`soft_peak` 和标准 `peak` 使用不同类型/字段并在结果中标记。
- **大模型显存**：流式 reduction、分片 ownership 和只保存请求通道，不默认保留全场历史。

## 11. 完成定义

当且仅当以下条件同时满足，SAR 项目才可标记完成：公共 API 与 `Scene + Simulation + Result` 一致；点/质量平均/归一化结果有单位和 provenance；1 g/10 g 边界与部分体素经标准案例验证；功率守恒、三档网格收敛、单/多 GPU parity 和关键梯度通过；生产尺寸 benchmark 达到预注册性能预算；文档明确支持的标准 profile 与不支持范围；`FEATURE_LIST.md`、教程和 benchmark 结果随实现一并更新。API 类存在但无标准/独立参考，不能计为完成。
