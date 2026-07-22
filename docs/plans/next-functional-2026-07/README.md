# Maxwell 下一阶段功能开发路线

> 日期：2026-07-14  
> 状态：规划基线  
> 入口：本目录所有功能计划的唯一导航文档

## 1. 路线目标

本路线把当前正在进行的数值可信度收敛视为持续工程基线，不把它重复包装成下一阶段功能。下一阶段的目标是把 Maxwell 已有的 GPU-first、PyTorch-native 全波内核扩展成一组可以独立交付、独立验收的 RF 与工程电磁能力。

路线遵守以下不可变约束：

- 公共求解架构保持 `Scene -> Simulation -> Result`；
- 本目录的所有全波功能、运行时、multi-GPU、梯度和性能计划均为 FDTD-only，不包含频域全波求解器的功能扩展或对齐工作；
- 新功能通过一等场景对象、`compile_*` 编译层和结构化 Result 数据接入；
- 不恢复 `Scene.set_*`、`Scene.with_*` 或任何旧后端类入口；
- 运行时继续 GPU-first，不以 CPU fallback 作为兼容路径；
- 优化和可微能力继续使用原生 PyTorch tensor、parameter 与 autograd 工作流；
- 每个用户可见交付必须同步更新根目录 `FEATURE_LIST.md`；
- 每个计划只描述一个可独立决策的产品能力或基础设施工程，内部 phases 不再拆成多个计划文件。

## 2. 范围边界

### 2.1 本路线包含

| 编号 | 计划 | 定位 |
|---|---|---|
| 01 | [RF 工程基础闭环](01-rf-engineering-workflow.md) | 端口、负载、网络参数和天线指标的共同底座 |
| 02 | [多 GPU 执行基础设施](02-multi-gpu-execution.md) | 任务级并行与单次 FDTD 联合域分解 |
| 03 | [Touchstone 网络嵌入](03-touchstone-network-embedding.md) | SnP 导入、因果/无源状态空间和 FDTD 端口耦合 |
| 04 | [SPICE/MNA 瞬态协同](04-spice-mna-cosimulation.md) | FDTD 与电路状态的同步时间推进 |
| 05 | [非线性电路器件](05-nonlinear-circuit-devices.md) | 二极管、非线性电容、开关等器件模型 |
| 06 | [阵列、Active S 与 MIMO](06-array-active-s-mimo.md) | 端口/远场叠加、codebook 和系统级指标 |
| 07 | [Thin-wire 模型](07-thin-wire-model.md) | 亚网格导线、电缆和细长导体建模 |
| 08 | [Ferrite 材料](08-ferrite-materials.md) | 磁化铁氧体和旋磁张量更新 |
| 09 | [表面阻抗与金属粗糙度](09-surface-impedance-metal-roughness.md) | 通用 SIBC、宽带表面阻抗和粗糙铜模型 |
| 10 | [SAR](10-sar.md) | 局部/平均 SAR 与暴露功率评估 |
| 11 | [Bioheat](11-bioheat.md) | EM 损耗到组织温升的耦合工作流 |
| 12 | [静电与电容](12-electrostatics-capacitance.md) | 静电势、场、电荷和电容矩阵求解 |
| 13 | [ESD 与介电击穿](13-esd-dielectric-breakdown.md) | 标准激励、应力/耐压评估与确定性介电状态演化 |

### 2.2 暂缓

- EME；
- 通用 Heat 产品线、稳态导电、半导体载流子/漂移扩散等未单独列项的多物理求解器；Bioheat 内部 thermal core 和 electrostatic free charge 不受此条影响；
- MPI 跨节点大规模生产部署；
- 云任务、CAD、PCB 编辑、材料数据库 UI 和其他 `witwins-studio` 职责；
- 仅为追平竞品类名而新增、但不能形成端到端工作流的对象。

本目录中的 Bioheat、静电与 ESD 是已经明确命名的后续项目，不代表本轮立即并行实现；它们用于冻结边界、依赖和验收标准。未列出的其他多物理方向继续暂缓。

## 3. 两条一级主线

### 3.1 功能主线：RF 工程基础闭环

`01-rf-engineering-workflow.md` 是第一功能项目。它一次性定义并交付：

```text
Scene
  -> RF Port / Load
  -> FDTD
  -> V / I / accepted power
  -> S / Z / Y
  -> far field
  -> gain / realized gain / efficiency
```

Touchstone、SPICE、非线性电路、阵列和若干垂直能力均依赖该计划提供的端口、网络数据和功率语义。

### 3.2 基础设施主线：多 GPU 执行

`02-multi-gpu-execution.md` 与 RF 基础闭环并行启动，但不要求完整域分解先于所有功能完成。它包含两种不同执行模式：

1. **ensemble execution**：把 N-port 逐端口激励、参数扫描和独立场景分配到 GPU device pool；
2. **joint solve**：把一个超大 FDTD 网格按空间分片，在 GPU 间交换 Yee halo。

前者优先服务 RF 吞吐，后者提供显存容量和大型单场景扩展。两者共享设备发现、失败传播、统计和 Result 顺序契约，但不共享物理求解状态。

## 4. 依赖图

```text
01 Phase 2 NetworkData ──> 03 Touchstone embedding
01 Phase 1/2 ports ──> 04 SPICE/MNA ──> 05 nonlinear circuit devices
01 Phase 2 + Phase 4 antenna data ──> 06 array / active S / MIMO
01 Phase 4 shared PowerLossData ──┬──> 10 SAR
                                  └──> 11 Bioheat EM coupling

07 thin-wire ──┐
08 ferrite ────┼── can start independently; later RF integration gates depend on 01
09 surface ────┘

01 terminal + 04 circuit ──> 13 ESD Phase 1/2
12 Phase 2 electrostatic + 04 Phase 2 circuit ──> 13 ESD Phase 3+

02 multi-GPU execution
├── ensemble execution supports 01 / 03 / 06 sweeps
└── joint-solve shard contract is integrated by every runtime feature
```

多 GPU 是横向基础设施依赖，不改变功能项目的物理依赖。`02` 提供通用 device pool、execution topology、domain partition/result manifest 和 collective 合同；FDTD、Bioheat、静电等 runtime 各自定义 stencil/state specialization，不能复用 Yee 专用 halo schedule。功能第一版可以在单 GPU 上完成，但从设计评审起必须声明其分片所有权、通信量、归约方式和暂不支持的 distributed 组合。

## 5. 推荐交付波次

### Wave A：共同底座

- 01 RF 工程基础闭环；
- 02 多 GPU 的 ensemble、partition/transport contract 和标准 forward MVP。

进入门：当前可信度工作已为相关源/monitor 提供稳定基线。Wave A 内部并行；RF public contract 与通用 execution/partition contract 分别评审。

Wave A 完成后，后续计划不得自行发明第二套端口、网络数据、设备池或 shard ownership。

### Wave B：网络与系统能力

- 03 Touchstone 网络嵌入；
- 04 SPICE/MNA 瞬态协同；
- 06 阵列、Active S 与 MIMO。

进入门：03 需要 01 Phase 2 的 `NetworkData`；04 需要 01 Phase 1/2 的端口与 RLC stamp；06 需要 01 Phase 2 的网络数据和 Phase 4 的绝对功率 `AntennaData`。03 和 06 可在各自进入门满足后并行；05 必须等 04 的线性 MNA 时间积分和耦合稳定后启动。

### Wave C：特殊 RF 数值模型

- 05 非线性电路器件；
- 07 Thin-wire；
- 08 Ferrite；
- 09 表面阻抗与金属粗糙度。

四项分别具有不同数值机制和验证基准，不合并为“高级材料”大项目。其启动顺序由目标市场决定，不制造虚假的串行依赖。

进入门：07/08/09 的 Phase 0-1 可与 Wave A 并行；它们只在端接、器件 S 参数、功率/天线验收 phase 依赖 01。05 的进入门是 04 Phase 2 的强耦合稳定。

### Wave D：行业与准静态能力

- 10 SAR；
- 11 Bioheat；
- 12 静电与电容；
- 13 ESD 与介电击穿。

进入门：10 与 11 并列消费 01 Phase 4 的共享 `PowerLossData`，Bioheat 不依赖 SAR 质量平均/VOP；若复用实现，则只依赖 10 Phase 1 的损耗数据，不依赖整个 SAR 项目。13 Phase 1/2 依赖 RF terminal，Phase 3 才同时依赖 12 的静电场/电荷与 04 的电路耦合。Wave D 在这些进入门满足前只维护设计计划，不承诺近期实现。

## 6. 所有计划的统一开发门

以下是完整门集合。每个 phase 按其目标证据等级累进适用，而不是要求 E0 原型提前满足 E3 production 的全部条件：

1. **API 门**：公开对象遵守 `Scene + Simulation + Result`，参数单位、方向、符号和失败模式明确；
2. **编译门**：场景对象只通过 `compile_*` 进入 solver state，不在 public object 中隐藏可变运行时；
3. **数值门**：至少有解析解、独立求解器、标准数据或守恒律中的两类证据；
4. **收敛门**：涉及 PDE/几何离散的能力必须报告空间、时间或模型阶数收敛；
5. **GPU 门**：正常推进步无新增 CPU round-trip、动态 device allocation 或隐式全局同步；
6. **PyTorch 门**：tensor 输入保留 device/dtype/梯度 lineage；不支持的梯度组合在 prepare 前明确失败；
7. **多 GPU 门**：声明 local ownership、halo/collective、结果归约和单卡 parity；
8. **Result 门**：结果是结构化、带坐标/单位/参考定义的 PyTorch-native 数据，不只是无约束字典；
9. **回归门**：针对性测试、公共 API 测试和相关 benchmark 通过；
10. **文档门**：`FEATURE_LIST.md`、示例、known limitations 和 benchmark 证据同步。

## 7. 验收证据等级

| 等级 | 含义 | 最低证据 |
|---|---|---|
| E0 | API prototype | 构造/验证测试；不宣称物理可用 |
| E1 | numerical prototype | 单一解析或独立参考、基本网格收敛 |
| E2 | engineering preview | 多场景参考、守恒/无源性、失败门、GPU profiling |
| E3 | production | 组合矩阵、性能边界、持久化、multi-GPU/梯度声明、公开 benchmark |

累进规则：E0 满足 API/编译/schema/失败校验；E1 在 E0 上增加单一独立参考和基本收敛；E2 再增加多场景数值证据、守恒/无源性、GPU profiling、Result/persistence 和 unsupported guards；E3 再增加组合矩阵、命名硬件性能边界、distributed/gradient 支持声明与公开 benchmark。对某一功能物理上不适用的门必须写明 `not applicable` 及原因，不能静默跳过。

计划中的 phase exit gate 必须写明达到哪个证据等级。只有 E2 及以上能力才能作为默认公开工作流；E0/E1 必须显式标记 experimental。

## 8. 状态管理

每份计划顶部维护：

- `Status`: proposed / active / blocked / complete；
- `Target evidence`: E0–E3；
- `Depends on`；
- `Owner modules`；
- 最近一次架构决策日期。

入口文档只维护项目级依赖和交付波次。详细实现状态、未完成项和验收证据留在对应计划中，避免 README 演变成第二份实现计划。

## 9. 路线完成定义

本路线不是要求 13 个项目同时完成。它在以下条件下完成其“规划基线”职责：

- 每个项目都有明确功能范围、非目标、架构、phases 和 exit gates；
- 依赖关系不存在循环或重复公共数据模型；
- RF 和多 GPU 两条一级主线可以独立启动并在稳定契约处汇合；
- 后续项目可以由单独开发周期领取，而无需重新决定公共架构；
- 暂缓方向没有被隐式塞入当前 phase。
