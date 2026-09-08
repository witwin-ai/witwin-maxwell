# FDTD Batch 仿真：公共 API、共享运行时与 GPU 调度开发方案

状态：实施中；逐项进度见下表，未验收项不视为完成。  
日期：2026-09-08。  
源码核对基线：`8deaf46`。  
交付目标：在固定背景中批量评估局部变化的场景，降低重复显存占用和准备开销，提高 GPU 吞吐，并保持 PyTorch 原生求导与 `Scene -> Simulation -> Result` 契约。

本文定义目标 API、运行时协议、分阶段范围和发布门槛。目标契约尚未全部交付；当前可执行范围以下表和紧随其后的使用说明为准。阶段完成必须有代码、测试和测量证据，不能仅依据此文声称功能已提供。

### 实施进度（2026-09-08）

| 子阶段 | 状态 | 证据与剩余工作 |
| --- | --- | --- |
| P0.1 固定负载与 storage 审计工具 | 完成 | `python -m benchmark batch --inventory/--audit`；7 项测试通过；GPU 1 的 launch audit 完成，原生扩展构建与运行通过；单次审计不代表性能验收 |
| P0.2 并发基准工具 | 完成 | 顺序/进程/run_many 的冷/热执行、有序频谱对照、隔离显存测量、CLI 和报告更新；N=3 GPU 冒烟通过，N=10 小网格 2 次预热/5 次正式测量通过 |
| P0.3 完整基线矩阵验收 | 未完成 | 其它负载、N=1/100、完全相同输入、N=10 隔离显存以及原业务几何仍待测；[已保存证据](../assessments/fdtd-batch-implementation-progress.md) |
| P1.1 描述对象与只读声明规划 | 完成 | 43 项公共 Batch 测试；公共 API/单场景/ensemble 回归共 79 passed、3 skipped；两种入口、梯度视图、输出冲突、明确物理差异与稳定分组已覆盖 |
| P1.2 编译证明、精确 output allocation 与运行时对接 | 未完成 | P1.1 report 始终 unresolved，不依据声明摘要共享张量或伪报显存预算；源尺度求导的场景物化仍明确拒绝 |
| P2.1 局部 Yee 材料采样基础 | 完成 | 同一材料函数支持局部裁剪和差分 halo；六分量、均匀/非均匀网格、偏振子像素和密度梯度对照；相关 GPU 测试共 85 passed |
| P3.1 原生 batch 场寻址基础 | 完成 | 复用原 E/H 更新与 interior fast path；共享/逐 lane 系数、64 位偏移、active mask；94 项专项 GPU 测试逐位对照通过，86 项原有原生/CPML/Graph/伴随/调制回归通过 |
| P3.2 原生 CPML 与协同时间推进 | 完成 | dense/slab CPML 复用原公式；固定容量场槽、尾批、重置与换入；复用普通 solve 生命周期的协同推进；167 项 GPU 专项/回归通过 |
| 公共前向执行与结果生命周期子阶段 | 完成 | `prepare/run` 可执行；原生 B-lane 场/CPML、重复运行、A/B/A 参数、尾批、31 频点 monitor-only、独立 Result、流式关闭/取消和保存/加载；公共 Batch/单场景/Graph 回归 89 passed |
| 原生 eager 与既有基线对照 | 工具/测量完成，组合性能门未通过 | launch、N=B=10、2000 步、2 预热/5 正式；原生 2.342 s，十进程 0.903 s；隔离驱动显存增量 411.6/3175.6 MiB；频谱误差为 0。不能据此声称耗时不退化或完整业务验收 |
| P2/P3 其余、P4–P7 | 未完成 | 背景/局部编译复用、源/观察器向量化、精确 output allocation、byte admission、协调 Graph、多 GPU/Q=2、调优和 Batch 伴随仍待实现；不视为全方案或性能验收完成 |
| P8 | 未开始 | 独立研究门，依赖 P4 实测 |

环境检查：沙箱内 GPU 不可见；获准在沙箱外检查后发现两张 A6000。
GPU 0 存在其它计算任务。原生扩展需显式设置当前环境内的 `nvidia/cu13` 为 `CUDA_HOME`。
硬件可用性与独占性能测量窗口分别记录；不修改其它任务或系统驱动。

### 当前可执行范围

`Batch.prepare/run` 已不再是不可执行的占位接口。当前执行层为
`native_field_cohort_eager`：一张 GPU、一个固定容量 B 的执行槽，E/H 与
dense/slab CPML 使用原生批量 kernel；每个 lane 仍通过现有流程注入点源、
累积点/面/通量监视器并完成归一化。返回的每一项都是普通 `Result`，
不引用活的 solver 或场槽。关闭 context 或复用场槽不会改写旧结果。

```python
import witwin.maxwell as mw

# simulations contains ordinary mw.Simulation.fdtd(...) objects.
submission = mw.Simulation.batch(
    simulations,
    execution=mw.BatchExecution(
        devices=("cuda:0",), batch_size=2, graph="off",
    ),
    outputs=mw.BatchOutput(monitors=("probe",), fields=()),
)
with submission.prepare() as prepared:
    results = prepared.run()
    spectra = results.stack(lambda r: r.raw_monitor("probe")["components"]["Ez"])
    repeated = prepared.run()
```

输入列表若还改变源幅度，应显式声明 `BatchVariation(sources=("drive",))`；
模板方式用 `BatchParameters` 的 density/source scale 张量，
`prepared.run(parameters=updated)` 必须保留相同 schema 与 N。
默认 `batch_size="auto"` 暂时保守选择 1，并不执行吞吐调优。
`graph="auto"` 暂时选择 eager；explicit `required`、`graph_steps`、
`tune=True`、严格 `memory_limit_bytes`、多 GPU/Q=2 和可训练输入均明确拒绝。
这些限制不能解读为完整调度器已经实现。

当前每个换入的 cohort 仍全域编译材料，密集系数按 lane 存储；只对实际相等的
spacing/CPML profile 共享存储。并未获得“背景只编译一次”的验收。
监视器频谱请求不会自动启用全域 DFT；显式请求 E 场时，原有累积器仍内部保存
全部 E 分量，再导出所选分量，精确子集分配仍待实现。`plan()` 仍为 unresolved，
`prepared.report` 记录实际执行层和已知限制。显存与耗时优势必须经后续实测，
不能直接套用原始截图中的数字。

已完成的小网格对照表明：原生 eager 相比顺序执行快约 2.23 倍、相比十进程
节省约 87% 的采样驱动显存增量，但耗时为十进程的约 2.59 倍。
这主要证明了减少独立 context 的收益，尚未证明背景张量共享节省。
因此“能执行”不等于“已适合在显存与吞吐两个指标上替代十进程”；
后续源/观察器批处理、编译复用、Graph 和调度阶段仍是必要工作。
完整数据与测量限定见 [实施证据](../assessments/fdtd-batch-implementation-progress.md)。

## 1. 需求、设计决策与收益边界

典型输入：同一张 GPU 上评估 10 个设计，场景大小、网格、边界、固定背景和观测需求基本相同，仅局部材料或结构发生变化。目前每个设计由独立进程执行。

统一术语：

| 符号 | 含义 |
| --- | --- |
| N | 一次提交的设计总数，例如 10、100、1000 |
| B | 某张 GPU 的单个执行槽中同时推进的设计数，即 resident batch size |
| G | 使用的 GPU 数量 |
| Q | 每张 GPU 同时存在的执行槽数量，默认 1，调优上限 2 |
| K | 一个 Graph chunk 中推进的时间步数 |
| R | 声明允许变化的空间区域，包含编译影响范围 |
| F | 输出频点数量 |

核心决策：

1. 每张 GPU 默认由一个持久 worker 管理，一个进程管理所有 worker；设计不对应独立进程。
2. 静态编译结果按兼容组、设备共享；电磁场及所有随时间演化的状态按设计独立。
3. 真正的 batch 更新进入已有原生 CUDA kernel；不能把 Python 循环启动 B 个 solver 作为最终实现。
4. 默认用一个 batch 占满 GPU；是否使用双槽和复制重叠由测量决定，不把 stream 数量等同于并行加速。
5. 保留现有普通 `Simulation`；Batch 是这些仿真的编排和执行优化，不引入第二套物理场景模型。
6. 优先支持频谱监视器与端口输出；全域场是显式选择，必须计入输出驻留预算。
7. GPU 原生前向和伴随构成正式版本范围；不通过 CPU 求解、NumPy 转换或隐式 detach 完成 batch。
8. 先交付 FDTD Batch。固定背景 Green 矩阵预计算作为独立后续阶段，不能隐藏在 `batch_size` 调优里。

对显存的预期采用可审计模型：

```text
独立进程：M ≈ N × (context + static + dynamic + workspace) + retained_outputs
单卡 Batch：M ≈ context + shared_static
                 + Q × (B × (dynamic + variant_coefficients) + workspace + graph_pool)
                 + input_staging + retained_outputs + autograd_retained
```

普通 FDTD 中，局部变化产生的散射会传播到全域，因此不同设计的 E/H、PML 历史、色散状态无法一般性共享。B=N 时仍有 N 份场；B<N 时用分批降低峰值，全部完成时间需要测量。相同输入去重默认关闭，避免改变结果独立性、用户计数或梯度语义。

用户提供的 1.11/11.07 GiB 与参考 bank 的 24.1 GiB 仅作为待复现实验的历史记录，不作为当前源码的测量结果，也不直接转换为承诺的节省比例。

## 2. 当前实现与需要修改的边界

| 当前入口或模块 | 已有行为 | Batch 需要的变化 |
| --- | --- | --- |
| `simulation.py::Simulation.fdtd` | 一个 Scene 对应一次仿真 | 保持单场景入口，新增 batch 编排工厂 |
| `simulation.py::PreparedSimulation` | `_claim()` 限定只能执行一次 | 保持原契约；新增有明确重置协议的 PreparedBatch |
| `execution/ensemble.py::run_many` | 调度独立 Simulation；拒绝训练参数 | 保留一般异构任务用途；共享调度基础设施，新增同构 batch 执行路径 |
| `execution/executor.py` | 线程池；多 worker 时暂停 capture | 新增协调式 capture 生命周期；不能直接删除现有保护 |
| `execution/pool.py` | 设备 lease 与静态容量快照 | 扩展字节级预约和共享资源计费，支持后台执行槽 |
| `execution/capacity.py` | 基于粗粒度乘数的启发式估算 | 增加编译产物逐项清单、峰值生命周期估算和运行校准 |
| `fdtd/cuda/kernels/*.cu` | 三维 Yee 场更新 | 加 batch stride、变体系数寻址、active mask；复用原数值公式 |
| `fdtd/cuda/runtime/graph.py` | 固定地址 Graph，进程内 capture 保护 | 可协调的预捕获、执行槽绑定、尾批掩码与缓存释放 |
| `fdtd/boundary/cpml.py` | dense/slab ψ 存储 | 共享 profile，ψ 保留 batch 维度；不重复实现 slab |
| `fdtd/observers.py` | 监视器 GPU 累积状态 | 累积器批量化、采样描述共享、输出按需分配 |
| `fdtd/adjoint/` | 原生伴随与 checkpoint | 批量反向、输入版本保护、checkpoint 生命周期与预算 |

当前 `Simulation._execute_fdtd_solve()` 中，多频点且没有 lumped port 会启用全域 DFT，即使 `full_field_dft=False`。Batch 必须建立统一的 output plan，由实际请求决定累积器；容量估算与运行时必须消费同一个 plan，不能继续各自推断。

当前 `MaterialRegion.density` 必须是三维，`SceneModule.to_scene()` 返回一个普通 Scene。保持这两个契约，batch 维度属于参数集合与运行时，不能把四维 density 塞进单个 Scene。

相关在用文档：

- [多 GPU 执行基础设施](next-functional-2026-07/02-multi-gpu-execution.md)
- [FDTD 多 GPU 实现](fdtd-multi-gpu-implementation-plan.md)
- [CUDA Graph 执行规划](fdtd-cuda-graph-option-b.md)

旧文档中的实现状态可能滞后；实现时以当前代码和重新运行的测试为准。

## 3. 目标公共 API

### 3.1 API 对象和职责

| 对象 | 状态 | 职责 |
| --- | --- | --- |
| `Scene` / `SceneModule` | 已有 | 单个物理问题；模块最终生成普通 Scene |
| `Simulation` | 已有 | 单个仿真的方法、频率、时间、激励配置 |
| `Simulation.batch(...)` | 新增 | 组装 Simulation 集合或模板参数批次 |
| `BatchParameters` | 新增 | 带 leading batch dimension 的命名参数张量 |
| `BatchVariation` | 新增 | 声明允许变化的对象及结构移动包络 |
| `BatchExecution` | 新增 | GPU、容量、B、Q、Graph、分组和调度策略 |
| `BatchOutput` | 新增 | 输出范围、结果目的地和持久化预算 |
| `BatchSimulation` | 新增 | 可检查的编排对象；不是新物理 solver |
| `PreparedBatch` | 新增 | 拥有共享编译结果、槽和 Graph 的运行上下文 |
| `BatchPlanReport` | 新增 | 兼容分组、逐项显存、执行方式及限制 |
| `BatchResult` | 新增 | 复用 `ResultSequence` 语义，索引返回普通 Result |
| `BatchItem` | 新增 | 流式完成项：index、variant_id、Result 或 failure、record |
| `BatchResultIterator` | 新增 | 有界、可关闭、支持上下文管理的 BatchItem 迭代器 |

### 3.2 标准入口：已有 Simulation 列表

下面示例中的 `scenes`、`frequencies`、`design_envelope` 由用户现有场景构建代码提供。`design_envelope` 使用共享核心几何对象，其范围覆盖所有允许的结构变化。

```python
import witwin.maxwell as mw

simulations = [
    mw.Simulation.fdtd(
        scene,
        frequencies=frequencies,
        run_time=mw.TimeConfig(time_steps=20000),
    )
    for scene in scenes
]

batch = mw.Simulation.batch(
    simulations,
    variant_ids=[f"design-{i}" for i in range(len(simulations))],
    variation=mw.BatchVariation(
        material_regions=("design",),
        structures={"patch": design_envelope},
    ),
    execution=mw.BatchExecution(
        devices=("cuda:0",),
        batch_size="auto",
        max_batch_size=10,
        memory_limit_bytes=8 * 1024**3,
        graph="auto",
    ),
    outputs=mw.BatchOutput(
        monitors=("near_field",),
        ports=("feed",),
        fields=(),
    ),
)

with batch.prepare() as prepared:
    report = prepared.report
    results = prepared.run()

first = results[0]
feed = first.port("feed")
```

`report` 是结构化对象；其 `to_dict()`/`to_markdown()` 可用于日志和评估文档。结果独立拥有输出存储，退出上下文后仍可读取。

### 3.3 高效入口：模板与参数张量

```python
import torch
import witwin.maxwell as mw

# template is a normal Simulation with a named MaterialRegion.
rho = torch.rand(10, 32, 24, 1, device="cuda:0")
parameters = mw.BatchParameters(
    region_densities={"design": rho},
)

batch = mw.Simulation.batch(
    template,
    parameters=parameters,
    execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2),
    outputs=mw.BatchOutput(monitors=("probe",), fields=()),
)

with batch.prepare() as prepared:
    result_a = prepared.run()
    result_b = prepared.run(
        parameters=mw.BatchParameters(region_densities={"design": 1.0 - rho}),
    )
```

模板路径使固定背景仅编译一次。每个逻辑变体仍能通过 `batch.scene(i)` 获取普通 Scene，密度为 `rho[i]`；仅显式调用时物化，不预先生成 N 份全域材料张量。`scene(i)` 返回独立容器，不能修改模板对象；参数张量的共享关系遵循 PyTorch 语义。

一个 `BatchParameters` 可同时提供多个 region 和源幅度系数，所有 leading dimension 必须一致：

```python
parameters = mw.BatchParameters(
    region_densities={"design_a": rho_a, "design_b": rho_b},
    source_scales={"drive": amplitudes},
)
```

`source_scales[name]` 是乘在模板源激励上的无量纲实标量，shape `[N]`；模板已有相位和波形保持其语义。不把复数幅度隐式解释为新的实数时域源。其它源参数、几何移动及材料类型切换通过 Simulation 列表表达，在后续 capability 阶段开放。

### 3.4 规范化签名

以下签名是目标协议，不是可直接粘贴执行的完整 Python 模块：

```text
Simulation.batch(
    simulations: Sequence[Simulation] | Simulation,
    *,
    parameters: BatchParameters | None = None,
    variant_ids: Sequence[str] | None = None,
    variation: BatchVariation | None = None,
    execution: BatchExecution,
    outputs: BatchOutput | None = None,
) -> BatchSimulation

BatchParameters(
    *,
    region_densities: Mapping[str, torch.Tensor] | None = None,
    source_scales: Mapping[str, torch.Tensor] | None = None,
)

BatchVariation(
    *,
    material_regions: tuple[str, ...] = (),
    structures: Mapping[str, Geometry] | None = None,
    sources: tuple[str, ...] = (),
)

BatchSimulation.plan() -> BatchPlanReport
BatchSimulation.scene(index: int) -> Scene
BatchSimulation.prepare() -> PreparedBatch
BatchSimulation.run() -> BatchResult
BatchSimulation.iter_results(*, order="completion") -> BatchResultIterator

PreparedBatch.report: BatchPlanReport
PreparedBatch.run(*, parameters: BatchParameters | None = None) -> BatchResult
PreparedBatch.iter_results(
    *, parameters: BatchParameters | None = None, order="completion"
) -> BatchResultIterator
PreparedBatch.request_cancel() -> None
PreparedBatch.close() -> None

BatchResult.stack(
    selector: Callable[[Result], torch.Tensor], *, device=None
) -> torch.Tensor
BatchResult.save(directory: str | Path) -> None
BatchResult.load(
    directory: str | Path, *, scenes: Sequence[Scene], map_location="cpu"
) -> BatchResult
```

输入规则：

- Simulation 序列不能同时传入 `parameters`；单个 Simulation 模板必须提供非空 `parameters`。
- 序列非空；ID 必须非空且唯一，默认采用稳定的十进制索引字符串。
- 每个 region density 为 `[N, nx, ny, nz]`，后三维与模板对应区域一致；值域由 MaterialRegion 验证。
- 不接受隐式 batch 广播。不变参数留在模板；单一输入张量内部的 `expand`/切片视图允许，但重叠写入禁止。
- 参数允许非连续输入；prepare/run 边界规范化一次，报告拷贝字节数，保留梯度链。
- 目标名称不存在、同名对象歧义、变更超出声明包络、参数 batch 长度不同，均在执行前给出带对象路径的错误。
- `variation=None` 的列表路径使用保守完整编译比较，只共享被证明相同的对象；不能猜测某片区域不变。
- 模板参数路径从命名 region 和 source 自动推导变化范围；额外传入 variation 时需与推导结果一致。
- `SceneModule` 通过已有 Simulation 物化逻辑接入；不调用 `.item()` 或 NumPy 将可训练几何参数断开。
- 默认执行设备来自 `BatchExecution`，不修改输入 Scene 的 `device`；逐项 Result 的 Scene 记录实际设备。

### 3.5 BatchExecution 完整配置

```text
BatchExecution(
    *,
    devices: tuple[str, ...],
    batch_size: int | Literal["auto"] = "auto",
    max_batch_size: int = 32,
    memory_limit_bytes: int | Mapping[str, int] | None = None,
    reserve_bytes: int = 512 * 1024**2,
    grouping: Literal["strict", "compatible"] = "strict",
    scheduling: Literal["static", "throughput"] = "throughput",
    slots_per_device: int | Literal["auto"] = 1,
    graph: Literal["auto", "required", "off"] = "auto",
    graph_steps: int | Literal["auto"] = "auto",
    tune: bool = False,
    fail_fast: bool = True,
)
```

| 参数 | 精确定义 |
| --- | --- |
| devices | 去重且显式编号的 CUDA 设备；禁止 CPU 求解设备；异构 GPU 可用但分别规划 |
| batch_size | 每槽逻辑最大驻留数；正整数为严格请求，容量不足报错；auto 才能缩小 |
| max_batch_size | auto 搜索上限；显式 batch_size 不得超过此值 |
| memory_limit_bytes | Batch 管理的增量 live allocation 预算，含静态、槽、Graph、输入 staging、输出与保存的梯度状态；不含调用前已有用户张量和上下文，但容量检查仍扣除它们实际占用 |
| reserve_bytes | 在当前可用显存中留出的安全余量；不宣称能阻止外部进程抢占 |
| grouping | strict 要求一个兼容组；compatible 显式允许分成多个真正 batch 组，无隐式独立进程回退 |
| scheduling | static 按报告固定分配；throughput 按完成速率和可用字节动态领取整批任务，保持返回索引 |
| slots_per_device | 允许 1、2、auto；auto 必须 tune=True，否则输入报错；两个槽拥有独立动态状态和 Graph |
| graph | required 不满足则 prepare 失败；auto 可选择原生 batched eager 并报告原因；off 完全不捕获 |
| graph_steps | 正整数或 auto；auto 初始 K=min(32, 最短检查间隔)，尾段掩码，不影响有效步数 |
| tune | 用隔离状态做有限候选调优；关闭时只做容量规划并使用保守默认，不伪称已找到最快 B |
| fail_fast | 遇错停止派发新批次；已经提交的 GPU chunk 在安全边界结束，故障语义见第 9 节 |

graph 配置归 BatchExecution 所有。包装后的 `Simulation.config.cuda_graph` 不再决定 batch capture；报告列出实际策略。频率、边界、时间步长等物理参数不能被执行器覆盖。

`plan()` 只进行只读验证和估算，不分配完整 solver；自适应网格无法廉价解析时报告 `unresolved` 与原因。`prepare()` 解析网格、完成精确计费、分配和 capture，返回 resolved report。不能让 dry plan 偷偷物化 N 份 GPU 场。

### 3.6 输出、结果与流式消费

```text
BatchOutput(
    *,
    monitors: tuple[str, ...] | Literal["all"] = "all",
    ports: tuple[str, ...] | Literal["all"] = "all",
    fields: tuple[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], ...] = (),
    materials: bool = False,
    result_device: Literal["local", "cpu"] | str = "local",
    max_pending_results: int = 2,
)
```

- `fields` 表示请求频点上的全域频域分量；时域采样通过 Scene 的时间监视器表达。E/H 插值和 Yee 时间错位沿用单仿真约定。
- 默认选取 Scene 明确声明的所有监视器与端口，不自动附加全域 DFT。`materials=False` 不持有全域编译材料副本。
- `outputs=None` 等价于默认 BatchOutput；如果子 Simulation 明确要求 `full_field_dft=True`，则报输出冲突并提示显式选择 fields，防止静默丢弃需求。
- `fields=()` 的 Result 仍是普通 Result；访问未保存的字段给出明确错误，不隐式重跑或补分配。
- `result_device="local"` 保留在执行设备；显式 CUDA 目标只迁移所选输出，不汇聚全域运行状态。
- `result_device="cpu"` 仅在无梯度运行中支持，作为显式结果导出，不是 CPU solver fallback；D2H 采用固定容量 pinned buffer。
- `.run()` 返回全部独立结果，必须预算 N 份输出，不能声称总显存只依赖 B。
- `.iter_results()` 不在内部累计完整 Result，只保留固定上限队列；队列满时停止提交后续批次。消费者自己持有的结果仍会占用显存。
- `order="completion"` 默认避免慢设计堵塞；`order="input"` 用有界重排队列，必要时降低提交量，不无限缓存。
- 流式迭代器是上下文管理对象，提前退出必须 close；顶层 `BatchSimulation.iter_results()` 的 close 同时关闭其内部 PreparedBatch。
- 每份输出必须脱离可重用槽存储；禁止把下一批会覆盖的 view 作为公开结果。多份 Result 可共享只读 metadata 和不可变几何。
- `Result.solver` 和 `raw_output` 不保留槽引用；prepared_scene 仅保留结果访问确实需要的不可变坐标与描述。Result.scene 的变体参数采用该次运行的值快照，更新 rho 或模板不能改写旧结果的场景含义；快照存储进入输出预算，autograd 原始输入另由 tape 持有。
- `BatchResult[i]` 返回普通 Result，失败时抛带 index/ID 的异常；`.entries/.records/.failures` 沿用 ResultSequence 的非抛异常检查方式。
- `stack(selector)` 对每项调用 selector，shape/dtype 必须一致；不同设备时要求显式 device，拷贝保留 autograd。任一项失败时不静默跳过。
- `save()` 使用批次 manifest + 普通 Result 文件，记录顺序、失败、配置摘要和 schema；不会保存活的 autograd 图。沿用现有 Result.save/load 不持久化 Scene 的契约，load 必须提供与 manifest 的顺序/ID 对应的 scenes，并核对配置摘要。失败槽不要求有结果文件；禁止序列化调度 callable。

```python
with batch.prepare() as prepared:
    with prepared.iter_results(order="completion") as items:
        for item in items:
            item.raise_for_error()
            item.result.save(output_dir / f"{item.index:06d}.pt")
            del item
```

### 3.7 PreparedBatch 生命周期和可重复执行

状态机：`NEW -> PREPARING -> READY -> RUNNING -> READY -> CLOSED`；不可恢复设备错误进入 `FAILED`，只能 close。

1. `BatchSimulation.run()` 是创建 PreparedBatch、run、close 的便利编排；内部不能破坏返回 Result 的存储。
2. 同一 PreparedBatch 不可重入，不允许并发 run/iter_results；不同 PreparedBatch 由全局调度注册表协调资源。
   `request_cancel()` 是唯一允许从其它线程调用的运行控制方法：只设置取消标记，在 chunk 边界处理，不强行终止 CUDA kernel。对 READY/CLOSED 调用为空操作；close 期间请求取消并等待本对象事件。
3. 模板模式允许用相同参数 schema 和 N 的新值重复执行。N、shape、dtype、频率或物理结构变化需重新 prepare；避免静默 recapture。
4. 列表模式重复 run 使用相同输入快照；更新列表需创建新 BatchSimulation。
5. prepare 后不允许原地修改场景结构、边界或共享材料；版本不匹配在 dispatch 前报错。模板参数值的更新经 `run(parameters=...)` 或无活跃执行时的合法 PyTorch 参数更新提供。
6. 每次 run 的输入版本固定到该次执行完成；前向保存的参数在 backward 前被原地修改时，按 PyTorch 版本检查报错。
7. `close()` 幂等，等待自身提交的事件，释放共享缓存引用、槽和 Graph。已有可求导结果通过独立 ExecutionTape 持有反向所需资源，因此 close 后结果仍可 backward；存活 tape 的显存不会假装已释放。
8. 常规显存归还 allocator 缓存，不在每批调用 `empty_cache()`。实际 live bytes 与 reserved bytes 分开报告。
9. 重复 run 前重新 admission，计入此前返回但仍存活的结果和 tape；首次 prepare 成功不意味着以后任意数量的保留结果都能放入相同预算。

### 3.8 BatchPlanReport 与完成项协议

`BatchPlanReport` 为不可变记录，最少提供以下字段；`to_dict()` 仅包含可序列化标量、列表、映射，不包含 tensor 内容或可执行对象：

```text
schema_version, resolved, variant_count, variant_ids
groups[]:
  group_id, variant_indices, signature_digest, capability_status
  shared_artifacts[], output_plan, unresolved_reasons[]
devices[]:
  device, group_id, batch_size, slots, graph_steps, graph_mode
  estimated_peak_bytes, memory_breakdown, reserved_headroom_bytes
  tuned, candidate_measurements[], rejected_candidates[]
input_copy_bytes, output_bytes, retained_tape_bytes
warnings[], unsupported_features[], configuration_digest
```

`BatchItem` 字段为 `index: int`、`variant_id: str`、`result: Result | None`、`failure: DistributedFailure | None`、`record: ExecutionRecord`。result 与 failure 恰有一个非空；`raise_for_error()` 成功时返回 None，失败时抛该项异常。迭代器 `.close()` 幂等，完成和失败项都占据原始 index；取消项也可通过最终 `.records` 查看。`BatchResult.records` 是最终记录，PreparedBatch.report 是规划快照，不在运行中被线程原地修改。

## 4. 兼容分组与静态数据共享

### 4.1 两层键

`ExecutionSignature` 决定能否进入同一个 batched kernel/Graph：

- solver 方法与原生 kernel 版本；设备能力、数值 dtype 与实数/复数模式。
- 解析后的各 Yee 分量 shape、网格坐标、间距和布局；不能只比较 Domain 大小。
- 六面 BoundaryKind、周期/Bloch 相位、PML 厚度与配置。
- 实际 dt、时间积分方案、终止上限、shutoff 检查计划和频谱采样定义。
- 材料模型类别、色散 pole 数与布局、各向异性结构、非线性或时变机制。
- 激励注入类型、数量、布局和辅助系统拓扑。
- 监视器采样布局、频点顺序、端口参考和后处理规范。
- variation 区域布局、伴随模式、checkpoint 计划、输出 plan。

`StaticArtifactKey` 决定具体哪些只读张量可共用：网格、PML profile、固定背景系数、源采样模板等可以分别命中缓存。同一个 ExecutionSignature 并不证明材料值相同。

材质变化可能改变稳定性条件和实际 dt。默认按实际 dt 分组；strict 模式报错，不自动减小 dt 或重采样源来凑 batch。用户需要同组时，应显式配置对所有变体都安全的 TimeConfig.dt。

自适应网格必须先冻结成全批统一解析网格；若几何变体产生不同网格，strict 报错或 compatible 分组。禁止把 shape 相同但坐标不同的网格误判为共享。

### 4.2 共享证明与缓存失效

- 模板模式对命名 region/源参数构造结构化变更集，背景由模板身份、结构版本和 tensor 版本共同确定。
- 列表模式以编译结果验证声明之外相同；共享不是仅凭名称、Python id 或 shape 推断。
- 编译比较可以采用 GPU 比较/摘要并在 prepare 汇总一次；不能为了哈希把全域材料复制到 CPU。摘要命中仍需可信 provenance 或精确比较来保证正确性。
- 用户声明的结构包络需扩展为编译影响区域：体素化、subpixel averaging、density filter、projection、Yee 插值和必要 stencil halo 都要包含。
- 区域重叠按现有 Scene 材料优先级合并，不能在 batch 路径改变重叠覆盖顺序。
- 固定背景 trainable 参数不能复用已经 detach 的编译值；数值存储共享与梯度计算图共享是两回事。
- 缓存由 PreparedBatch 生命周期持有，跨 PreparedBatch 可通过有上限的资源注册表复用；禁止无上限全局强引用。
- 修改背景、网格、dt、边界或输出布局使相应缓存失效；更新局部密度仅重编译区域系数，不重建背景或 Graph。

## 5. GPU 数据布局与数值更新

### 5.1 状态所有权

| 类别 | 布局/所有权 | 重用与重置 |
| --- | --- | --- |
| 网格差分距离、边界模式、静态索引 | 每组每设备一份 | 只读 |
| CPML profile / kappa / a / b | 每组每设备一份 | 只读；dt 属于键 |
| 固定材料和更新系数 | 一份背景，按分量存储 | 只读；标量均匀系数保留标量快路 |
| 局部变体系数 | `[B, R_component]` 或区域致密块 | 每批覆盖；含编译 halo |
| Ex/Ey/Ez/Hx/Hy/Hz | 每分量 `[B, nx_c, ny_c, nz_c]` | 各自真实 Yee shape；整批清零/初始化 |
| CPML ψ | 每方向 `[B, slab_shape...]` | 每次运行独立重置 |
| 色散/ADE/非线性状态 | `[B, active_material_state...]` | 按模型注册 reset 与 checkpoint |
| TFSF/入射辅助场 | 初版独立 `[B, ...]` | 不因入射配置相同擅自共享可写状态 |
| 电路、网络、导线、端口状态 | 按变体独立 | 后续 capability 验收后开放 |
| DFT/时域监视器 | `[B, F, sample_shape...]` | 仅分配 output plan 所需分量 |
| 计数器、active/failed mask、能量统计 | `[B]` 或明确共享的时钟 | 重置包括峰值能量、计数和归一化 |

`B=1` 与现有原生单仿真共享数值函数。最终只维护一份离散更新公式；不能拷贝整套 `electric.cu` 为第二个 batch solver。

### 5.2 kernel launch 与系数读取

- 首版采用 component-major + batch-major + contiguous spatial 布局，保持空间相邻线程连续访问。
- launch 把 batch 编号与空间 tile 编号组合，使用 64 位总元素数和地址偏移；检查 grid 维度上限，不能简单把 B 乘进已有 z 维后溢出。
- 所有边界访问先得到 batch 内局部坐标；周期 wrap 与 Bloch 相移绝不能跨到另一设计。
- 每槽 active mask 屏蔽尾批和已结束设计；所有参与该步的源、边界、观测与归一化 kernel 都必须使用同一有效性语义。
- 区域系数首版采用一个共享空间映射：背景位置直接读取共享系数，设计区域位置读取当前 lane 的系数。映射和 branch 成本需要测量。
- 后续可把空间分成不重叠背景 tiles 和区域 tiles，分别启动 kernel 降低分支；同一个分量、同一时间步的同一单元只能写一次。
- 不用“先更新背景、再用另一个 kernel 覆盖区域”的粗暴实现，否则可能重复推进有状态材料或错误使用更新后的场。
- 区域过大时允许编译器选择每变体致密系数，报告 `coefficient_layout=dense` 及原因；仍是真正 batched CUDA 路径。
- dtype 不自动降精度；复数场按现有实虚布局演进；新寻址保持 LibTorch Stable ABI 和所有受支持设备架构的构建要求。

### 5.3 单步依赖与重置协议

以现有 solver 的完整 update schedule 为唯一依据，逐操作标注 read/write/dependency；磁场更新、电场更新、材料辅助状态、源校正、边界校正、耦合端口及观察器按原时间错位执行。不能为了融合而调整物理顺序。

引入内部 `StateDescriptor`，每个状态声明：shape、dtype、owner、reset action、capture mutation、checkpoint inclusion、adjoint requirement 和 estimated bytes。descriptor 用于校验与计划生成，热循环不遍历 Python 对象。

每批执行：

```text
等待 slot 可用
→ 绑定 variant IDs 与参数版本
→ 编译/装载局部系数
→ reset E/H、ψ、材料状态、源辅助场、端口、电路、监视器、时钟
→ 应用本次 initial condition
→ 推进有效步数
→ 归一化与 finalize 到独立输出缓冲
→ 发布完成事件
→ 输出不再引用 slot 后归还 slot
```

任何新增物理状态没有 descriptor 或 reset/checkpoint 实现，prepare 必须拒绝 batch；不能因为遗漏注册就默认它是只读。

## 6. GPU 底层调度

### 6.1 分层与线程模型

```text
BatchSimulation / PreparedBatch
        ↓
Compatibility groups + OutputPlan + MemoryPlan
        ↓
Process-wide resource/capture coordinator
        ↓
DeviceWorker(cuda:0)           DeviceWorker(cuda:1)
  shared artifacts              shared artifacts
  slot 0 [B variants]           slot 0 [B variants]
  optional slot 1               optional slot 1
        ↓                             ↓
  native batched kernels        native batched kernels
        ↓                             ↓
       ordered Result / bounded result stream
```

默认一个进程、每卡一个提交线程。线程承担批次级提交，时间步工作由 Graph/native kernels 完成。不要求线程数等于 B，不通过 fork 继承 CUDA context。空间分域的多 GPU FDTD 是另一种调度维度，首版禁止在 batch 子 Simulation 中同时指定 `parallel`。

### 6.2 Streams 与 Events

每设备一个 compute stream；Q=2 时每槽独立 compute stream。另有有界 input/output copy stream，仅在存在实际复制需求时创建。

```text
producer/current stream ── input_ready ──> staging/compile
staging/compile          ── params_ready ─> compute(slot)
compute(slot)            ── output_ready ─> output copy/finalize
output copy/finalize     ── slot_reusable > next batch
output ready event      ── wait on caller stream before tensor consumption
```

原生扩展必须使用目标设备的 PyTorch current stream，不硬编码 default stream。跨 stream 张量用事件和 allocator 生命周期登记保护；只有需要 host 读取完成状态、导出或返回完成契约时做对应事件等待，不在每个时间步做全设备 synchronize。

Q=2 的主要候选用途是把输入/结果搬运与推进重叠。若单槽已受显存带宽限制，不应为双槽付出两份动态状态而降低吞吐。发布默认保持 Q=1；auto 只在容量充足且重复测量有收益时选择 2。

### 6.3 Graph 捕获与重播

Graph 依赖固定地址，capture 私有池及输出引用也影响释放时机；跨 Graph 共享池有执行顺序和并发约束。[PyTorch CUDA Graph 内存说明](https://docs.pytorch.org/docs/2.14/notes/cuda.html#graph-memory-management)

采用以下设计约束：

1. prepare 先加载扩展、初始化库、分配全部固定缓冲，再做 warmup、reset、capture、reset。
2. 捕获前后状态通过 descriptor 恢复；非零 initial condition 不能当作零场处理，也不能把 warmup/capture 产生的场或监视器数据带进正式仿真。
3. 所有 worker 在 capture epoch 进入前到达安全边界并暂停 CUDA 提交，逐设备逐槽捕获，结束后再并发 replay。只锁 capture 调用而允许其它线程继续分配或同步不够。
4. 注册表协调库内普通 run_many、Batch 和独立 Simulation 的捕获；外部用户线程的 CUDA 工作无法完全控制，文档明确 prepare 需应用提供静默初始化窗口。required 失败就报错；auto 只有已知“不支持捕获”可改用 batched eager。
5. capture 错误不得 blanket catch 后吞掉：越界、设备错误、空 Graph、数值发散不是正常 eager 降级理由。
6. 不采用 relaxed capture mode 来绕开真实依赖错误。capture 中无 `.item()`、D2H、Python 动态分配、模块加载或与当前 epoch 无关的 CUDA 提交。
7. 时间、源幅度、DFT 相位、active mask 和停止标记来自固定地址 device tensor；不能捕获 Python 标量时间而冻结波形。
8. Graph cache key 包含 signature、device、slot、B capacity、K、output/checkpoint layout 与代码版本。参数数值不进入 key，以原地写入固定输入槽更新。
9. 每槽最多保留正式 K-chunk 和必要的尾步 Graph；不为每个 variant 创建一个 Graph。
10. 默认 Graph 私有池不跨并发槽共享。仅能对已证明串行且输出生命周期满足条件的图共享 pool，并单独验收。
11. 尾批填充 inactive lanes；尾 chunk 每步用目标步数掩码。active 之外的 lane 不推进任何状态，不贡献 DFT、收敛统计或梯度。
12. 正式执行期间不 recapture。新 signature 先停止派发、排空本库 worker，再进入新 capture epoch；首版默认组切换时串行完成此过程。
13. checkpoint 边界与观察器采样点不可因 K 改变而移动。规划器选择可分段的 K，或在图内固定步骤保存到预分配 checkpoint 槽；动态新增 checkpoint 分配必须放在 capture/replay 外并重新计费。

CUDA Graph 表达的是带依赖的操作图，合法执行顺序由依赖约束决定。[CUDA Graph 执行模型](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) 本方案依靠事件与读写依赖保证正确性，不依赖观察到的某次 stream 排序。

### 6.4 自动 batch 选择与公平调度

容量规划步骤：

1. 读取每设备当前 free/total、allocator 已保留但可复用的空间和本库资源预约，避免把 reserved bytes 重复计费；采用可校验的保守下界处理碎片。
2. 用解析后的 tensor manifest 枚举共享部分、B 相关状态、Graph 峰值、编译临时量、输出和 tape，形成分阶段峰值。
3. 可用预算同时受用户 memory_limit、物理可用空间和 reserve 约束；每次 admission 再查，不只依赖 prepare 时快照。
4. auto 从有限集合 B={1,2,4,8,...,max_batch_size} 及 N 尾端候选中选择可放入值；tune=False 优先保守容量方案，报告未测吞吐。
5. tune=True 在隔离状态上测准备开销、代表性 chunk 和 finalize，包含 CPML/源/监视器；不得只测裸 curl kernel。参数和初始场恢复后才开始正式结果。
6. 调优最多 6 个 B 候选、每候选 3 次短测；K 候选 {1,8,32}，Q 候选 {1,2} 分层筛选，避免笛卡尔积搜索。
7. 对吞吐相差不超过 5% 的方案选择显存更低的方案；冷启动总任务很少时计入 tune/capture 成本。
8. 多 GPU 以待执行整批为调度单位，优先复用已有 shared artifacts，再依据实测完成率派发；不迁移正在推进的 E/H。
9. 不同 GPU 可选不同 B、K；结果顺序由提交 index 决定。static 模式记录固定映射用于复现。
10. 多个库内任务用轮转和有界连续 chunk 防止饥饿；stream priority 仅为提示，不承诺硬实时抢占运行中的 kernel。

### 6.5 提前结束、失败 lane 和尾批

shutoff 按每个设计独立峰值能量和与单仿真相同的检查规则决定。kernel 在逻辑检查点更新 active；host 每 chunk 最多读取小型状态摘要。K 超过检查间隔时，检查必须位于 Graph 内对应步骤，不能延迟到 chunk 末改变有效积分时间。

初版不向仍有 active lane 的槽填入新设计。停止 lane 保留状态至整个槽 finalize，避免混合不同时间原点和归一化窗口。结束时间差异很大时，planner 以运行计划分组，降低尾部浪费；动态 lane refill 属于后续优化，不是首版依赖。

## 7. 显存与结果预算

MemoryPlan 至少列出以下字节项及 peak phase：

- 进程 context/库开销的设备外部观测值，与本库增量 tensor 占用分列。
- 场、PML profile、PML ψ、材料值、更新系数、PEC mask、区域索引与参数。
- 源波形/延迟表、端口矩阵、辅助物理状态。
- 频谱/时域监视器、用户请求全场输出、归一化暂存。
- Graph warmup/capture 临时量、私有池和缓存。
- 输入 staging、异设备拷贝和结果重排队列。
- 梯度 checkpoint、反向 workspace、输入版本快照及尚未 backward 的 tape。

容量规则：

1. 先给 N 份用户已持有输入与保留输出留空间，再选择 B；不能仅对 solver 内部估算。
2. 模板路径按 batch 编译局部密度，避免分配 `[N, full_grid]`。若用户自己传入 N 份全域材料，报告 input bytes，不能声称优化器已消除这些输入。
3. 无全域输出时，全域 DFT 分配必须为零；增加 F 只增加请求的频域观测存储及相应采样表。
4. `.run()` 预计结果放不下时 prepare 报容量错误，提示减少输出或使用流式结果，不默默搬到 CPU。
5. 固定 B、Q 与有限输出队列下，纯前向流式运行的内部 live GPU state 应与 N 无关；参数集合和用户保留对象独立计费。
6. CPU 输出、文件写入和进度显示都必须在 Graph 外；单独报告导出时间与等待时间。

PyTorch 的 allocated 与 reserved 指标含义不同，外部显存观测还包含上下文等开销。[PyTorch 显存管理说明](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management) 基准同时报告三类指标，不用 `empty_cache()` 后的一个数字替代峰值。

## 8. PyTorch 训练与伴随设计

### 8.1 用户接口

```python
rho = torch.nn.Parameter(torch.rand(10, 32, 24, 1, device="cuda:0"))
optimizer = torch.optim.Adam([rho], lr=1e-2)

batch = mw.Simulation.batch(
    template,
    parameters=mw.BatchParameters(region_densities={"design": rho}),
    execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2),
    outputs=mw.BatchOutput(monitors=("probe",), fields=()),
)

with batch.prepare() as prepared:
    optimizer.zero_grad()
    results = prepared.run()
    values = results.stack(lambda result: objective(result))
    loss = values.mean()
    loss.backward()
    optimizer.step()
```

`objective(result)` 是用户提供、返回标量 torch.Tensor 的函数；不是新增的库 API。执行和导数仍通过普通 Result 中的张量建立关系。

### 8.2 明确的内存语义

`.run()` 后再对 N 个结果一起 backward 时，保存的 checkpoint/tape 可以随 N 增长。Batch 不能同时承诺保留任意计算图、任意延迟 backward 和 O(B) 训练显存。

需要限制训练峰值时采用同步流式反向：

```python
with batch.prepare() as prepared:
    optimizer.zero_grad()
    with prepared.iter_results(order="input") as items:
        for item in items:
            item.raise_for_error()
            (objective(item.result) / len(rho)).backward()
            del item
    optimizer.step()
```

该模式初版 `max_pending_results=1`，yield 期间不推进下一 microbatch，已 yield 的一批最多保留 B 份 tape。逐 Result backward 可多次进入同一个 microbatch 的伴随；所有该批 tape 释放后才复用反向资源。需要更高效率时后续增加按 microbatch 消费接口，不能改变逐 Result 的梯度语义。

为支持该契约，各 Result 的 solver autograd 节点拥有独立的 tape 生命周期，可以引用只读共享 checkpoint storage，但不共用一个会在首次 backward 后释放所有 saved tensors 的多输出 Function 节点。单项 backward 只求该 lane 的 VJP；聚合 loss 时由执行层在合法情况下合并反向提交。这项设计需同时通过逐项与一次聚合 backward 测试，不能依靠用户给 solver 节点添加 retain_graph=True 补救。

对 `rho` 这样的叶参数切片，可逐项累积梯度。若多个变体来自同一次共享的非叶 PyTorch 前处理图，逐项 backward 可能需要 `retain_graph=True`；用户需在每 microbatch 重算前处理，或保留该图并计入内存。本库不隐式 retain_graph，也不伪称任何 SceneModule 都能按上述示例实现固定训练显存。

### 8.3 反向实现与限制

- 复用现有 CUDA adjoint 公式，加入 batch stride；禁止保留每一步 Python autograd 节点导致 O(B × time_steps × grid) 存储。
- checkpoint 捕获完整动态状态，包含 ψ、材料历史、激励辅助状态及已支持的耦合子系统。
- 每次 forward 产生独立 ExecutionTape，保存输入引用和版本、参数映射、stop steps、checkpoint plan、device 与完成事件。
- 槽复用前将反向必需状态转交 tape；不能保留会被下一批覆盖的裸指针。
- 独立参数的梯度不得跨 lane 混合；共享 trainable 背景参数按所有 lane 的贡献求和，不能只保留最后一个设计的贡献。
- normalization、采样窗口、有效步数和源幅度梯度与单仿真一致。提前停止视作离散执行决策，不对停止时刻求导，并报告实际步数。
- 一阶 backward 为正式范围；高阶梯度/JVP/vmap 全流程未验收前显式拒绝，不能从“PyTorch-native”推断全部支持。
- `retain_graph=True` 保留 tape 并计费，重复 backward 的生命周期需测试；close 不使活跃 tape 失效。
- 初版多 GPU 梯度允许输出显式 gather 到目标 CUDA 后求和，通过可微 copy 回传参数源设备；不插入 NumPy/CPU 中转，不自动开启分布式进程组。
- 背景共享参数的跨 GPU 梯度按固定设备顺序汇总；动态图归约顺序造成的浮点差异纳入容差。DDP 的应用级梯度归约另由用户控制。
- 在梯度能力阶段完成前，任何 requires_grad 输入都在 prepare 明确拒绝；不能自动 detach 后以成功返回。

## 9. 错误、取消与可观测性

批次级配置错误（名称、维度、signature、未知 capability）在分配动态场前报错，并列出 variant_id 与字段差异。

执行失败分类扩展现有 FailureKind，避免另造不兼容错误体系：

| 分类 | 行为 |
| --- | --- |
| capacity | 提前估算不足或分配失败；带每项估算及设备可用值 |
| numerical | GPU 检测到该 lane 的非有限场/发散；标失败并屏蔽其后更新 |
| runtime | CUDA/扩展执行错误；当前槽全部受影响项标失败 |
| cancelled | fail_fast 或流式 close 后未开始的项；保留原 index |

对于可恢复的分配 OOM：只有 auto 配置、尚未开始该 microbatch、未产生副作用或 tape 时，允许释放当前未发布资源并降低 B，最多一次重规划；Graph 重建必须进入协调 capture epoch。固定 B 不静默缩小。非法内存访问等设备错误使 worker 失效，不重试污染 context。

GPU kernel 出错时通常无法可靠定位单个 lane，错误归属整个受影响 microbatch；只有显式 device-side 数值检测才能逐 lane 归属。其它 GPU 是否继续由 fail_fast 决定。

报告和 Result metadata 必须包含：

- variant_id、输入顺序、组 ID、解析网格/配置摘要、有效时间步数。
- 执行设备、B/Q/K、coefficient layout、CPML mode、graph on/off 与原因。
- 共享张量字节数、每 lane 字节数、输入/输出/tape 字节数、估算峰值与观测峰值。
- prepare/compile/warmup/capture/tune/solve/finalize/export/wait 各阶段时间。
- H2D/D2H/P2P 字节数、Graph capture 次数、kernel/replay 次数、停止/失败状态。
- dtype、kernel build、运行设备信息、代码版本、采样和归一化配置。

GPU 时间只通过对应设备 Events 测量；端到端时间包含真实等待与结果就绪。性能报告明确是否独占设备，不把排队时间隐藏成 solver 加速。

## 10. 功能范围与能力矩阵

每个 capability 的 batch eager、Graph、adjoint 分别有验证状态，不允许一个总开关绕过未实现路径。

| 能力 | 首个支持阶段 | 正式要求 |
| --- | --- | --- |
| 固定均匀/显式非均匀网格、实数无色散材料 | P2/P3 | 原生 batched eager 与 Graph |
| density region、局部 PEC 图案 | P2/P3 | 区域编译、halo 和场更新等价 |
| 点源、点/面频谱监视器 | P2/P3 | 源时间和归一化逐 lane 正确 |
| CPML slab、PEC/PMC/周期边界 | P2/P3 | ψ 独立，边界不串 lane |
| 多频点、Flux/闭合面及 lumped port | P4 | 用户 31 频点局部图案验收场景 |
| 复数 Bloch、TFSF、mode source/monitor | P7 | 完整源/边界状态与频谱 parity |
| 色散、各向异性、非线性、时变材料 | P7 | 独立状态注册和稳定性；按模型逐项开放 |
| 单/多 GPU ensemble batch | P5 | 每设备独立 B，共享背景每卡一份 |
| density 和源尺度一阶梯度 | P6 | 单 GPU 与多 GPU 梯度、流式 backward |
| trainable shared background | P6 | 跨 lane/device 梯度求和 |
| 电路、网络、导线、表面阻抗、初始条件 | P7 | 逐类与现有单仿真完整路径对照 |
| 单个变体跨 GPU 空间分域 + batch | 后续独立方案 | P0-P7 显式拒绝 |
| FDFD 多变体、Green bank | P8 研究门 | 不阻塞 FDTD Batch 正式发布 |

P7 按子能力发布：某项未验收时报告明确不支持，不将整个 P7 自动标为完成。功能覆盖报告需从验证器生成，避免文档与运行时代码漂移。

## 11. 开发 phases 与逐阶段验收

依赖主线：`P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7`。P8 在 P4 数据具备后独立评估。以下工作量是单人专注工程时间的初始估计，不是交付日期承诺；P0 后根据实际 kernel 覆盖更新。

### P0：基线与显存审计（3–5 个工作日）

交付：建立固定场景 benchmark，记录 N=1/10/100、独立顺序/独立进程/现有 run_many；输出 tensor manifest、Graph/DFT/CPML 分配报告和稳定的比较脚本。

验收：

- 同一份物理模型、网格、dt、终止步数、dtype、源与监视器，生成可重跑基线。
- 至少三个负载：launch 开销主导小网格、带宽主导大网格、局部图案 + 31 频点端口/监视器。
- 分开测“完全相同的十份输入”与“十个真实不同的设计”；相同输入用例关闭去重。
- 对 1.11/11.07 GiB 历史数字标出能否复现及差异来源，不用历史数字替代数据。
- memory manifest 覆盖所有主要 live tensor 和持有者，列出与设备外部峰值的剩余差额。
- 确认多频点全域 DFT 触发条件及 slab/dense ψ 的实际选择。
- 记录独立进程并发是否真的缩短十设计 makespan。

### P1：API、编译计划与输出契约（5–8 个工作日）

交付：新增公共描述对象、Simulation.batch、兼容验证、结构化 plan、统一 output plan；运行能力尚未具备的配置明确失败。

验收：

- 第 3 节两种构造方式、参数错误、ID、schema、输出冲突均有公共 API 测试。
- strict 对网格坐标、dt、边界、频点、布局不一致给出具体差异；compatible 给出稳定分组。
- plan 不分配 N 份全域场；未解析网格明确标记 unresolved。
- output plan 与容量估算共用；fields=() 时多频点不能隐式规划全域 DFT。
- 单 Scene、SceneModule 与现有 PreparedSimulation 契约不变。
- FEATURE_LIST 只登记已实现的实验性规划接口及限制，不能登记尚不存在的执行能力。

### P2：共享编译与可重置执行槽（8–12 个工作日）

交付：共享只读 artifacts、局部参数编译、StateDescriptor、B=1 的持久执行槽、结果独立存储与有界流式输出。

验收：

- 同模板 100 次参数更新，背景编译计数为 1；每次只更新声明区域及其 halo。
- A/B/A 交替执行与三次独立运行通过第 12 节数值门槛，含 ψ 和监视器状态污染探测。
- 可训练输入在 P6 前明确拒绝，不能返回 detached 成功结果。
- 生命周期测试覆盖 run/close、异常、提前关闭迭代器、旧结果在槽复用后保持数值不变。
- 固定 B=1、Q=1，N 从 10 到 1000 时内部 live GPU 峰值增长不超过 max(2%, 16 MiB)，排除明确计费的输入、用户保留结果及 allocator reserved。
- MaterialRegion filter/subpixel halo 内外均与全域重编译一致；变化越界被拒绝。

### P3：原生 batched CUDA 与协调 Graph（12–18 个工作日）

交付：电磁更新、CPML、源和基础观察器 batch 化；设备参数缓冲、K-chunk Graph、尾批与 active mask；扩展构建更新。

验收：

- B=1/2/3/4/8/10 与逐项独立运行对照，N=10、B=4 的尾批和非 K 整数倍步数正确。
- 存在 CPML 活跃场、周期边界穿越、非零初始值的 reset/capture 测试，不能只测全零场。
- 每次核心时间更新的 kernel 数由操作类别决定，不随 B 线性增加；profile 中不能出现 B 个独立 solver 的热循环。
- 稳态 Graph 推进无 H2D/D2H、`.item()` 或分配；仅 chunk 边界允许有界状态摘要。
- 同 schema 连续 100 个 microbatch，capture 次数不随 microbatch 数增加。
- 两个 PreparedBatch 与原 run_many 交错使用不发生 capture 竞争；关闭一个对象不影响另一个仍活跃的共享资源。
- 内存检查工具无跨 lane 越界、use-after-free、未初始化读取；所有扩展声明支持的构建目标通过 ABI/加载检查。
- B=1 热推进性能相比现有单 solver 中位数回退不超过 5%；如超过，先优化再进入下一阶段。

### P4：多频点/RF 输出与自动容量、吞吐规划（8–12 个工作日）

交付：31 频点端口/Flux/闭合面输出、显存预约、B/K 调优、导出背压、完整性能报告。这是满足原始十设计前向需求的首个交付点。

验收：

- 31 频点只请求端口/局部监视器时，全域 DFT allocation 为 0。
- 所有频点幅相、端口归一化、功率和能量指标满足第 12 节标准。
- 自动预算在注入容量竞争时不超额派发；固定 B 容量不足提前失败；auto 降低 B 的重试有上限且不重复发布结果。
- 冷/热场景分别比较单进程顺序、十独立进程、现有 run_many、新 Batch；报告 N 项全部就绪的 makespan。
- tune=True 在预设候选中的热吞吐达到实测最佳值的 90% 以上；候选和误差范围写入报告。
- launch 主导场景目标为热吞吐至少 1.5 倍单进程顺序基线；带宽主导场景不承诺线性加速，热吞吐回退不超过 5%。
- 在预先固定的静态数据占比高场景，B=N 的 tensor 峰值至少下降 20%；同时满足第 12 节共享斜率门槛。比例未达标时不能把测例换小掩盖问题。
- 十设计业务场景必须提供一个通过数值门槛、显存更低且 makespan 不劣于十进程超过 5% 的配置，才能宣称“同时节省显存与时间”；做不到则如实交付显存/吞吐 Pareto 曲线并标明此业务目标未通过。

### P5：多 GPU、双槽与任务级资源协调（8–12 个工作日）

交付：每卡 worker、不同 B 的容量分配、跨设备参数/结果传输、静态/动态调度、可选双槽。

验收：

- 1/2/4 GPU 配置输出与单卡一致；硬件不足的配置标记 blocked，不把 mock 测试当真实多卡验收。
- 同型号双 GPU、足够多均匀设计时，热吞吐至少达到单 GPU 的 1.7 倍；包含真实输出但单独报告初始化成本。
- 异构/不均匀剩余显存下，各卡选择合法 B，不将所有场景预分配到协调卡。
- 独立变体推进期间 P2P/collective 字节为 0；只允许输入分发、结果汇聚和必要的后续梯度传输。
- 双槽选中时相比单槽重复测量收益至少 5%，否则 auto 保持单槽；两个槽的私有池和输出互不覆盖。
- 取消、某卡 OOM、device runtime 故障和慢消费者不导致死锁；每个输入都有最终完成/失败/取消记录。
- 当前非默认设备 Graph 绑定测试与既有 ensemble 测试通过。

### P6：一阶伴随与训练生命周期（12–20 个工作日）

交付：batch adjoint、区域参数 pullback、源尺度梯度、共享背景梯度归约、tape 生命周期、流式梯度累积。

验收：

- 独立逐项梯度、batch 梯度和有限差分方向导数满足第 12 节门槛。
- 独立输入 lane 的交叉梯度为零；共享参数梯度等于独立各项贡献求和。
- CPML ψ 活跃、复杂归一化、尾批、不同 stop steps、连续优化迭代都有端到端证据。
- optimizer 更新后输出与新参数匹配，背景/Graph 不错误使用上轮系数；forward 后非法原地修改在 backward 检测到。
- close 后仍可 backward；丢弃未 backward Result 后 tape 和预约字节可释放；重复 backward/retain_graph 符合约定。
- 流式叶参数案例 N=10/100/1000、固定 B 下，除输入/梯度本身外的内部 live GPU 峰值满足 max(5%, 32 MiB) 增长门槛。
- 列表、模板和 SceneModule 的 native PyTorch 链路均通过；没有 NumPy、CPU solver 或隐式 detach。
- 多 GPU gather/梯度回传与单 GPU 梯度一致，caller stream 中 optimizer 无事件竞争。

### P7：高级物理覆盖、发布与维护（按能力分项估算）

交付：第 10 节高级模型逐项接入；能力矩阵、公共示例、文档、基准及迁移说明完整。

验收：

- 每个新增模型必须注册状态并具备 eager/Graph/reset/checkpoint/adjoint 的明确支持级别。
- Bloch/TFSF/ADE/各向异性等使用非平凡场景，与现有单仿真数值和梯度测试对照，不仅检查“能运行”。
- RF 电路/网络/导线状态保持原求解时序和端口功率约定；模型组合没有验证时明确拒绝。
- 所有已有公共 API、边界、源、材料、RF 和梯度相关回归通过；B=1 性能回退仍在门槛内。
- FEATURE_LIST 与实际 capability 一致；没有遗留复制 kernel、兼容 shim 或实验路径冒充正式实现。
- 文档包含“十设计局部变化”“大批次流式输出”“模块式优化”“多 GPU batch”四个可运行示例。
- 完成阶段的状态与证据写入 docs/memory、docs/assessments；整个方案完成或被替代后归档到 docs/archive/plans。

### P8：固定背景响应预计算研究门（不纳入 P0-P7 交付承诺）

参考方法以预计算的设计区域数值 Green 矩阵复用固定背景响应，局部少量自由度变化可进一步使用低秩更新。[论文的方法与预计算部分](https://www.nature.com/articles/s41467-026-69477-y)

研究内容：用 Batch FDTD 并行生成响应矩阵列；在 GPU 上保存各频点 bank；小批量求解变体；按需要预计算端口/监视器投影。保持每个结果能映射到普通 Result 的输出契约，但需要独立 solver 能力设计，不在本方案中虚构已支持的 `method`。

继续投入的验收门：

- 固定网格和背景、线性材料条件明确；变化范围、频率和源/观测空间定义完整。
- 估算并实测 bank `O(F × m²)`、构建峰值、每候选分解工作区，m 使用 Yee 自由度而非像素数。
- 同离散模型的普通 FDTD 与 bank 解结果通过数值门槛；全场重建和有限时间频谱误差单独报告。
- 按 `T_precompute + N × T_candidate` 与 Batch FDTD 总成本比较，实测得到摊销临界 N。
- 少量局部低秩修改与任意大幅变化分别测量，不把前者的复杂度和毫秒性能推广给后者。
- 如果只评估 10 个变体不能摊回预计算成本，保留 Batch FDTD 为默认业务方案。

## 12. 统一验收标准与实验协议

### 12.1 正确性门槛

同离散网格、dt、有效步数、dtype、源、材料、监视器与归一化的独立单仿真为主 oracle，另用解析/守恒关系防止双方共享实现错误。

定义 `rel_l2(x,y)=||x-y||₂/max(||y||₂, floor)`；floor 按用例物理参考尺度预先固定。不同物理量不能统一套用一个无单位 atol。

| 指标 | 初始硬门槛 |
| --- | --- |
| 非衰减至噪声的场/频谱 rel_l2 | ≤ 1e-4 |
| 局部最大绝对误差 | ≤ 1e-4 × 预先声明的同量纲参考幅值 |
| 无量纲 S 参数复数绝对误差 | ≤ 1e-4 |
| 相位差 | 幅值高于预先定义噪声阈值时 ≤ 0.1° |
| 端口功率/Flux 的相对误差 | 非零参考量上 ≤ 1e-3；近零量用同量纲绝对阈值 |
| batch vs 独立一阶梯度 rel_l2 | ≤ 1e-3，且方向余弦 ≥ 0.9999（非零梯度） |
| 有限差分方向导数 | 至少 3 个步长寻找收敛区间，最佳稳定区间相对误差 ≤ 1e-2 |
| lane 交叉梯度 | 独立问题应为精确零，或有书面解释的浮点零阈值 |

完全相同操作序列的点更新优先要求 bitwise equality；发生归约顺序变化时使用上述容差。高 Q/长时间/强非线性场景若需要不同阈值，必须在测量前用收敛分析形成独立验收标准，不能为通过测试事后放宽。

必须包括以下反例：只修改一个 lane 其它 lane 输出不变；改变 PML/dt/网格坐标导致拒绝共享；区域外材料改变不命中背景缓存；A/B/A reset；尾批 poisoned inactive lane 不污染有效 lane；相同 shape 不同频点不能误用 Graph。

### 12.2 共享与峰值显存门槛

- 同设备同兼容组的共享张量 storage identity 唯一，并使用 storage 去重计费。
- E/H、ψ、累积器的有效写入区间在各 lane 之间不重叠。
- 用 B=1/2/4/8 拟合 `M_live(B)=intercept+B×slope`；slope 不含 N 份静态背景，实测与清单预测偏差 ≤ 10%。
- capture 峰值与稳态峰值分别验收；如果输出/临时 workspace 使内存非线性，需要在 plan 中显式列出，不强行拟合直线。
- 用户预算内计划应使预测 live 峰值不低估实测超过 10%；安全余量不足时下调 B，而非赌一次分配成功。
- 总设备峰值、PyTorch max_memory_allocated/max_memory_reserved、按对象清单 live bytes 全部保存。
- 100 次重复 run/close 后无单调 live allocation 增长；缓存有明确上限且清理后引用可释放。

### 12.3 性能测量协议

- 独占 GPU 窗口固定驱动、CUDA/PyTorch/扩展版本、设备与功耗状态，记录而不默认修改系统配置。
- 每个配置至少 2 次预热、5 次正式测量；交错运行基线与候选，报告中位数、范围和原始数据。
- 冷启动包含进程、模块加载、编译、capture/tune 和最终输出就绪；热运行复用 PreparedBatch。
- 延迟指标包括首结果时间、N 项全部就绪 makespan、variants/s、cell-updates/s；不能拿每设计摊销时间冒充单设计延迟。
- 输入数据和输出目的地一致，不能一方保存全场另一方只保存端口。
- 对进程基线记录整张 GPU 峰值，而不是把各进程不同时刻的峰值相加当作同步峰值。
- 大网格重点看带宽与状态流量；小网格重点看 launch/Graph 重播次数。改进结论只对测过的负载范围有效。

### 12.4 测试与基准落点

新增测试按现有类型组织，不新建一套平行测试系统：

| 路径 | 覆盖 |
| --- | --- |
| `tests/api/public/test_batch_simulation.py` | 两种 API、结果、错误和生命周期 |
| `tests/materials/compiler/test_batch_material_compiler.py` | 背景共享、region halo、失效与训练图 |
| `tests/fdtd/test_batch_state.py` | 独立状态、reset、尾批和 shutoff |
| `tests/fdtd/cuda/test_cuda_batch.py` | native 寻址、边界、dtype 和非默认 stream |
| `tests/fdtd/test_batch_cuda_graph.py` | 捕获协调、参数更新、私有池、输出生命周期 |
| `tests/fdtd/multi_gpu/test_batch_execution.py` | 分配、吞吐、异构显存、失败与背压 |
| `tests/gradients/test_fdtd_batch_adjoint.py` | 梯度 parity、tape 和共享参数 |
| `tests/validation/benchmark/test_batch_benchmark.py` | 基准配置与报告完整性 |

扩展现有 `python -m benchmark` 注册表，新增行为命名的 batch 场景及执行模式；CLI 参数在 P0 明确并写入帮助。输出仍使用 `benchmark/plots/`、`benchmark/RESULTS.md`，原始调度/显存记录放同一基准输出体系。不要创建只在个人脚本中可运行的验证入口。

所有 Python 命令、脚本和 pytest 必须在 `maxwell` conda 环境执行。阶段先运行相关 targeted tests；数值核心和 API 改动通过后运行受影响的更广回归。纯文档阶段不运行 GPU 数值测试冒充实现验收。

## 13. 文件与模块实施清单

| 文件/目录 | 实施任务 |
| --- | --- |
| `witwin/maxwell/__init__.py` | 只导出已交付的 Batch 公共对象 |
| `witwin/maxwell/simulation.py` | Simulation.batch 工厂、共享 output plan 接入 |
| `witwin/maxwell/scene.py` | 保留单场景契约，必要时增加 `compile_*` 命名的区域编译 helper |
| `witwin/maxwell/compiler/` | 共享 artifact 与区域编译；在主材料编译模块中扩展数值逻辑 |
| `witwin/maxwell/execution/batch.py` | 新增 API 描述对象及 BatchSimulation/PreparedBatch 编排 |
| `witwin/maxwell/execution/plan.py` | 兼容组、signature、execution report、output plan |
| `witwin/maxwell/execution/capacity.py` | tensor manifest 和 admission 峰值计算 |
| `witwin/maxwell/execution/pool.py` | 设备级字节预约、共享缓存计费、任务公平性 |
| `witwin/maxwell/execution/executor.py` | worker、槽、事件依赖、capture epoch 与流式背压 |
| `witwin/maxwell/execution/records.py` | BatchResult/BatchItem 与复用的失败记录 |
| `witwin/maxwell/fdtd/runtime/` | 显式状态所有权、reset、batch step 与时间控制 |
| `witwin/maxwell/fdtd/cuda/` | 现有 kernel 增加 batch 布局，更新声明、绑定与构建测试 |
| `witwin/maxwell/fdtd/cuda/runtime/graph.py` | 槽绑定 capture/replay、协调器和池生命周期 |
| `witwin/maxwell/fdtd/observers.py` | output plan 驱动的 batch 累积和 finalize |
| `witwin/maxwell/fdtd/adjoint/` | tape、checkpoint、native batched reverse 和 pullback |
| `witwin/maxwell/result.py` | 不持有 solver 槽的部分输出契约、持久化支持 |
| `FEATURE_LIST.md` | 每次交付新增实际用户能力时同步更新 |

文件拆分以实际职责为依据，保持已有核心模块为数值实现的唯一来源。`BatchSimulation` 只编排；不新增 `mw.FDTD`、`mw.FDFD` 或 Scene.set_/with_ 风格入口。

## 14. 发布判定与未决事项处理

P4 完成可发布“实验性单 GPU Batch 前向”，必须附支持能力矩阵；P5 完成开放多 GPU；P6 完成才开放训练参数。P7 每项高级能力单独通过后登记。

正式完成标准：API 示例可运行，原始十设计场景有可重跑证据，数值/显存/性能/生命周期门槛通过，硬件不足项有明确状态，FEATURE_LIST 与验证器一致。尚未测出的收益标记为目标，尚未支持的能力明确失败。

实施前 P0 需要冻结业务验收场景的四项信息：真实变化对象及包络、总评估数量与是否长期复用、所需输出、是否需要梯度。缺少用户业务场景时先完成通用三类基准和 API 开发，不把通用基准通过等同于业务验收通过。

本次交付仅为开发设计文档，不修改 FEATURE_LIST 中当前可用能力，不更改 solver 行为。后续实现每个用户可见能力时，必须在同一变更中更新 FEATURE_LIST。
