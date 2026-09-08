# FDTD Batch 性能瓶颈与正确性审计

日期：2026-09-08。审计起点：`1c1dc67`；确定性正确性修复：`d232245`。
范围：当前 `native_field_cohort_eager` 单 GPU、单执行槽前向实现。
完整项目状态仍是实施中，见 [开发方案](../plans/fdtd-batch-simulation-api-and-gpu-scheduling.md) 和 [已有验收证据](fdtd-batch-implementation-progress.md)。

## 1. 结论

存在明显改进空间，但三类瓶颈必须分别处理：

1. 小网格：Python 与 CUDA 提交开销主导，源和观察器仍按 lane 执行。31 频点场景中，观察器占 CPU profiling 区间的 **89–90%**。
2. 大网格：场更新占约 **99% 的已记录 GPU 工作时间**。主要优化目标转为场更新的数据访问和计算；目前没有硬件带宽计数器证据，不能宣称已经达到显存带宽极限。
3. N 超过驻留容量 B：每次重复运行仍重新构建 N 个 solver。短任务中，这个准备成本可以与时间推进同样重要。

显存也有独立于调度的改进空间：大网格的 2677.4 MiB 可达张量中，节点/边材料编译缓存约 **1407.3 MiB**；仅未启用的非对角、非线性和调制属性就物化了约 **353.1 MiB**。优化应同时审视普通材料编译器的重复表示和按需分配，而不只盯着 Batch 场槽。

正确性审计发现并修复了参数缓存错误和执行所有权竞态；另外补齐 inference tensor 的安全更新路径，并让未验证的结构包络明确拒绝执行。支持范围内的数值对照通过，并不代表完整 Batch 调度、梯度、所有高级物理或业务场景已经验收。

此前 2000 步 N=B=10 的独立基准中，Batch 为 **2.342 s**，十进程为 **0.903 s**；Batch 尚慢 **2.59 倍**。隔离测量的驱动显存增量分别为 **411.6 / 3175.6 MiB**。这证明减少进程上下文能节省显存，尚未证明背景材料共享，更未通过“显存降低且吞吐不退化”的组合门槛。

## 2. 测量方法与边界

- 所有 Python 在 `maxwell` conda 环境运行；PyTorch `2.13.0+cu130`，驱动 `595.71.05`，RTX A6000 48 GiB。
- 使用物理 GPU 1，映射为进程内 `cuda:0`。GPU 0 有其它计算任务，GPU 1 有桌面显存占用。各审计 GPU 工作串行执行；没有锁定整机 CPU、GPU 时钟或功耗。
- 固定基准的 16³、48³、128³ 是名义域网格；实际 Yee 数组包含边界/PML 扩展和分量错位。不能直接用名义格点数推算已分配字节。
- 初次探索每项预热 1 次、计时 5 次；仓库 CLI 复测预热 2 次、计时 5 次。计时包含 `run()`、结果检查和 GPU 就绪，不含 prepare。另行记录 prepare 时间。
- `cProfile` 定位 Python 热点，`torch.profiler` 给场、源、观察器、同步、导出添加区间。profiling 会显著拖慢主机提交，因此区间耗时不能替代无 profiler 的业务耗时。
- CUDA 活跃比例用 kernel/memcpy 等区间的并集除以设备时间轴跨度；排除跨越空闲段的用户 annotation。这个比例不是 SM occupancy，也不是内存带宽利用率。
- Graph 实验只捕获六个分量的场更新，保持源、观察器和主机时间控制不变，并验证输出频谱。它是独占当前进程的诊断实验，**没有交付公共 Batch Graph 调度器**。
- 新增短任务结果是定位实验。原业务几何、长时间谱收敛、独占机器性能矩阵仍待验收。

原始文件在 `benchmark/plots/batch/`：

| 文件 | 内容 |
| --- | --- |
| `audit-launch-profile.json` | CLI：名义 16³、B=10、F=1、128 步 |
| `audit-spectrum-profile.json` | CLI：名义 48³、B=10、F=31、64 步 |
| `audit-bandwidth-profile.json` | CLI：名义 128³、B=4、F=1、64 步 |
| `audit-bandwidth-storage-profile.json` | 补充同一大网格的完整去重 storage ownership 清单 |
| `audit-bandwidth-storage-summary.json` | 根据首次可达 owner 对唯一 storage 作互斥归类；另列未启用属性子集 |
| `audit-{launch,spectrum,bandwidth}-exploratory.json` | 第一次探索原始结果，保留复测差异 |
| `audit-capacity-n10.json` | N=10、B=1/2/4/10，每个容量预热后再运行三次 |
| `audit-correctness-before-fixes.json` | 修复前边界用例的原始失败证据 |
| `launch-n10-native-eager.json` | 先前的 2000 步、四模式吞吐和隔离显存对照 |

## 3. 时间推进瓶颈

### 3.1 CPU 提交与 GPU 工作应分开看

以下为 CLI profiling；CPU 占比的分母是当次 `audit.run` 区间。

| 场景 | 无 profiler 热运行中位数 | 场更新 CPU 区间 | 源 CPU 区间 | 观察器 CPU 区间 | GPU 活跃区间占设备跨度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16³ / B10 / F1 / T128 | 0.174 s | 14.6% | 26.2% | 39.2% | 4.5% |
| 48³ / B10 / F31 / T64 | 0.816 s | 2.2% | 3.6% | 90.3% | 5.1% |
| 128³ / B4 / F1 / T64 | 约 0.147 s | 约 18% | 约 13% | 约 17% | 约 92% |

初次探索对应热中位数为 0.103、0.492、0.146 s。小网格两轮差异明显，而已记录 GPU kernel 时间接近，符合主机提交受环境影响的特征；这不是定位某个其它进程或 CPU 频率原因的充分证据。

大网格中，GPU 场更新约 140 ms，总 GPU 工作约 141.6 ms。同步的 CPU 区间只是等待先前异步工作完成，不能简单认定删掉同步就能获得同等加速。源与观察器在 CPU 区间占用非零，但很大部分可与 GPU 场计算重叠，已经不在同样的关键路径上。

### 3.2 多频点观察器是最明确的热点

代码：`fdtd/observers.py::accumulate_observers`、`execution/batch_runtime.py::_solve_cohort`。

当前每一步先统一更新 B 个 lane 的场，然后逐 lane 恢复普通 solver 的 Python generator。每个 lane 的观察器又逐频点循环。

对于本次一个点、一个场分量的探针：

```text
场更新调用数       = 6 × T
源注入调用数       = B × T
点谱累积调用数     = B × F × T
F31、B10、T64      => 384 次场更新，640 次源注入，19,840 次点谱累积
```

频点从 1 增至 31，场更新 kernel 数量不变，观察器提交量增加 31 倍。31 频点 CLI profile 中，观察器 CPU 区间约 1323 ms，而关联 GPU 工作只有约 31.8 ms。主要损耗是大量很小的工作项经过 Python、参数处理、扩展调用和 CUDA 提交。

建议将固定布局的谱状态表示为 `[B, F, P]`，按分量/监视器布局批量 gather 和累积。复杂度中的数值工作仍需覆盖 B×F×P，但主机 launch 数量不应继续线性依赖 B×F。面与通量监视器必须保持 Yee 插值、E/H 半步相位、方向符号、窗和源归一化。

### 3.3 源与场调用本身仍有可削减开销

代码：`fdtd/excitation/temporal.py::apply_compiled_source_terms`、`fdtd/runtime/batch.py::NativeFieldSlot.advance`。

点源仍逐 lane 在 Python 计算波形、处理 patch 元数据并提交注入。场更新虽已经批量化，每一步仍重新构造六组 Python 参数列表，拼接字段名、查询 CPML 布局；原生入口还执行形状、设备和别名检查。

可以在 bind 阶段构建固定调用描述，预存 CPML 布局，统一生成 GPU 波形/权重并进行批量源注入。完整性检查应前移到固定描述建立时，并在重新 bind 时失效；不能为速度直接移除公开原生入口的安全检查。

### 3.4 只捕获场更新的 Graph 有收益上限

下表为无 profiler 的成对控制实验；“下降”是耗时减少比例，不是吞吐倍数。所有捕获实验探针误差为 0。

| 场景 | 初次控制 → Graph | 耗时下降 | CLI 复测控制 → Graph | 耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| 16³ / F1 | 0.1209 → 0.0951 s | 21.3% | 0.2118 → 0.1622 s | 23.4% |
| 48³ / F31 | 0.6190 → 0.6032 s | 2.6% | 1.0539 → 0.9562 s | 9.3% |
| 128³ / F1 | 0.14618 → 0.14570 s | 0.33% | 约 0.1483 → 0.1479 s | 约 0.28% |

多频点复测的单轮耗时波动很大，9.3% 不能视为稳定提升承诺。两轮实验都支持：只捕获场更新没有消除 B×F 的观察器提交；大网格收益接近测量噪声。要跨越十进程基线，优先优化源和观察器，再实现覆盖完整 K 步的 Graph，才有合理的关键路径依据。

## 4. 准备、换批与显存

### 4.1 PreparedBatch 只缓存当前 cohort

代码：`execution/batch_runtime.py::_load`。缓存键包含当前 indices 和参数身份/版本，只有一个已加载 cohort。后续 cohort 完整编译材料、构建 solver，再复制进固定场槽。

名义 16³、F1、T128、N10 的重复运行实测：

| 容量 B | 每次 run 的 solver 重建数 | run 中位数 | solver 构建时间中位数 |
| --- | ---: | ---: | ---: |
| 1 | 10 | 0.781 s | 0.240 s |
| 2 | 10 | 0.589 s | 0.237 s |
| 4 | 10 | 0.548 s | 0.256 s |
| 10 | 0 | 0.173 s | 0 |

构建计时包括材料编译及 solver 初始化，不能全部归因于材料采样。B4 下约 47% 的短任务时间落在该区间。这个诊断 sweep 只有三次测量且容量顺序固定，不能当自动调优结果。稳定事实是每轮重建次数，而不是某个容量的普适加速倍数。

默认 `batch_size="auto"` 实际取 1。因此仅切换到 Batch 默认设置，并不能获得全部驻留或消除重复编译。

### 4.2 当前共享范围有限

- E/H 和 CPML ψ 是每个变体的独立演化状态，需要随 B 增长；局部材料不同也可能最终改变整个空间的场。
- 当前只对编译后实际相等的网格间距和 CPML profile 共享。十二组更新系数仍按 lane 密集保存，普通 solver 的其它材料/辅助张量也保留。
- 当前 slot 根可达的去重 GPU storage，三类工作负载分别约 45.8、555.3、2677.4 MiB。这是可枚举张量的存储，包含通过 solver 引用到的对象，不等同于只看 `slot.buffers`，也不是驱动或 allocator 峰值。
- 大网格六个场的 slot buffer 合计约 232.8 MiB，十二组系数约 465.7 MiB，slab CPML ψ 约 27.3 MiB；其它可达存储还占相当部分，需要沿 ownership manifest 继续削减重复物化。
- monitor-only 路径没有全场 DFT；但请求单个 `Ez` 全场谱时，内部仍为全部 E 分量分配频谱工作区。
- `run()` 会持有全部 N 项结果；结果显存仍随 N 增长。流式消费可以释放已处理结果，但 `max_pending_results` 不是整个运行时的严格显存预算。

大网格完整 ownership 清单进一步将 2677.4 MiB 拆分如下。每个唯一 storage 只归入其第一个可达 owner，避免 views 重复计费：

| 归属 | MiB |
| --- | ---: |
| B-lane 场、CPML、系数 slot buffers | 725.8 |
| Scene 节点材料编译缓存 | 941.7 |
| Scene Yee 边材料编译缓存 | 465.7 |
| solver 的节点材料表示 | 78.5 |
| solver 的边材料表示 | 232.8 |
| 电/磁色散模板中保留的表示 | 232.8 |
| 其它 | 0.09 |

其中 `compiler/materials.py::_new_material_model` 无条件物化 `eps_offdiag_components`、`kerr_chi3/chi2/tpa_sigma` 和三个 modulation 数组；本次静态各向同性负载并未启用这些物理。它们合计 353.1 MiB，已包含在节点缓存一行中，**不能再次相加**。实际 `Ex` 槽形状为 `[4,136,137,137]`，这也确认名义 128³ 与含 PML 的真实分配不同。

这揭示了两种不同优化：按需要构建未启用属性；梳理节点、边采样、运行时材料和模板之间的可共享/可释放表示。后者涉及源系数、重跑、后处理和未来梯度的依赖，不能直接清空缓存就声称释放表中全部空间。此次审计没有修改这些数值/材料表示，表中数字是实际占用，不是已经获得的节省。

可以共享固定材料背景、恢复已验证的常量系数快路径，减少重复采样结果和不需要的谱工作区。但区域编译必须先证明实际影响范围，包括滤波/投影 halo、Yee 采样、与源 patch 的交叠及周期边界。当前不能凭几何包络直接共享背景。

### 4.3 GPU 调度尚有明确边界

已有独立 stream、事件依赖、结果存储所有权和同一 PreparedBatch 的非重入保护。尚无设备级字节预约、跨 PreparedBatch 的统一准入、协调 Graph capture、Q=2 槽重叠、多 GPU 分发和实测自动调优。

`reserve_bytes` 只检查当时的 free memory 是否高于阈值，不会为后续所有分配预留字节。新 solver 与旧驻留槽在换入阶段可以同时存活，prepare 峰值不能用运行中的槽大小代替。严格 `memory_limit_bytes` 目前明确拒绝。

增加 stream 或 Q 不会自动消除 Python 的逐频点提交；大网格也可能因争用而变慢。调度扩展应在单槽完整流水线和内存记账具备后开展。

## 5. 正确性发现与修复

| 等级 | 修复前问题与证据 | 当前处理 |
| --- | --- | --- |
| P1：静默错误结果 | 临时 source scale 被释放后，Python 复用了 tensor 的 `id`；缓存仍用 `(id, _version)` 命中旧参数。60 次更新中复现 26 次错谱，最大相对误差约 84.6% | 缓存持有输入张量强引用直到替换/关闭；回归验证临时对象生命周期及幅度线性关系 |
| P1：执行槽所有权失效 | 另一线程关闭执行中的 iterator，generator 抛出 `already executing`，但 finally 提前释放 prepared 锁。实测下一步仍执行时锁已解锁 | iterator 加独立操作锁；活跃 next 时拒绝 close/第二个 next。取消后等待 next 返回，再正常释放 |
| P2：关闭/启动竞态 | 输入验证期间 prepared 已关闭，验证结束后仍可建立新 iterator | 在取得 prepared 锁后再次检查 closed 状态 |
| P2：推理输入无法使用 | 真正的 inference tensor 没有 `_version`，prepare 直接抛异常 | 该类输入每次保守重编译；原位更新后结果正确 |
| P2：声明契约未落实 | 指向设计之外极小区域的结构变化包络仍被接受运行 | 编译影响证明完成前明确拒绝包络声明；省略包络仍可全域编译 |

所有权竞态的复现只检查了锁不变量，未故意同时启动两个 GPU 时间循环；没有把潜在竞态夸称为已经测出的数值污染。包络问题原先绕过的是声明校验，因为仍执行完整编译，不能据此声称此前所有物理结果都错误。

`_version` 无法识别绕过 PyTorch 的共享存储写入或 `.data` 修改。当前使用契约已明确：采用受版本跟踪的 PyTorch 操作，或者提交新的参数张量。输入 Scene 在 prepared 生命周期内保持不变。训练输入仍明确拒绝。

## 6. 验证覆盖与未证明事项

新增五项边界回归覆盖缓存身份、inference 更新、两个并发关闭边界和包络拒绝。新增四项数值回归覆盖：

- 非均匀网格，PEC、PMC、CPML；不满容量尾 lane。
- 3 频点、Hanning 窗、源归一化；E 全场谱和点、面、通量观察器，与独立运行比较。
- 完整场输出在再次 run 和关闭 context 后保持不变。
- 两个 lane 不同停止时间：一个脉冲衰减提前终止，另一个继续到计划步数；输出和停止步数与独立运行一致。

数值对照使用 `rtol=1e-4, atol=1e-7`，结果持有/重跑一致性使用严格相等。全场比较会规范化普通 Result 与 Batch 内部字段名大小写，逐一要求实际字段存在，避免缺失键被跳过而造成假通过。

相关既有覆盖还包括六个原生场分量、dense/slab CPML、31 频点、尾批/active mask、重置、A/B/A 输入、CPU 导出、结果持久化、失败取消及释放。下方完整命令结果为 **247 passed，0 failed**，耗时 156.75 s；一个既有 sparse tensor invariant 警告，未出现跳过项。

仍未证明：

1. 与同一数值内核的独立运行一致，不能排除共有离散误差；仍需要网格收敛、能量/反射、解析解或独立参考的业务验证。源幅度线性关系提供了一项不同于简单 kernel 对照的检查。
2. 本次环境中未运行设备内存 sanitizer，未做跨 GPU 架构矩阵，也未完成长期 N=1000 全输出寿命压力测试。
3. 现有普通单场景 adjoint 回归不能证明 Batch 梯度；Batch backward 尚未实现。
4. 高级物理、RF 端口、多 GPU 和完整 Graph 仍受显式能力门约束。后续 cohort 是换入前验证，并非 prepare 时已完成所有运行时证明。
5. 执行记录的 `wall_time_s` 从 cohort 开始计时，后续 lane 的记录还可能包含消费前一项的等待时间；不能拿它作为每设计纯求解时间。性能判定使用外部完整 makespan。
6. 保存/加载依赖调用方提供正确的 Scene；持久化场景指纹校验尚不能作为本阶段通过项。

## 7. 建议的后续开发顺序与验收门

以下是工程验收目标，不是本次已测出的优化收益；与主计划子阶段对应，每项独立提交。

| 顺序 | 实施内容 | 必须满足的验收标准 |
| --- | --- | --- |
| A，已交付 | 缓存与执行所有权修复 | 原先边界失败可复现；修复后专项及受影响 GPU 回归通过；FEATURE_LIST 与约束一致 |
| B，优先 | 向量化点源与频谱观察器；固定调用描述 | 固定布局主机 launch 数不随 B×F 增长；F=1/31、B=1/2/10、尾批、窗/归一化、E/H 相位和面/通量通过；完整输出就绪计时，禁止以删观察器换速度 |
| C | GPU 时间/频谱权重和覆盖完整 K 步的 Graph | 捕获/重播无逐步主机相位依赖；A/B/A、mask、尾块、取消、shutoff、非默认 stream 和旧 Result 生命周期通过；比较整个 run，包括 capture 摊销 |
| D0，可与 B 独立推进 | 未启用材料属性按需构建，清理重复材料表示 | 先核对所有读取方和 autograd 依赖；静态负载不再产生无用的整域属性张量；以去重 ownership 和 allocator 峰值验证，单场景高级物理回归同样通过 |
| D，可独立推进 | 背景材料共享与区域系数重编译 | 先完成编译影响范围证明；区域与全域系数、源耦合系数和最终场一致；N>B 重跑消除无必要的全 solver 构建；共享有明确容量/引用计数且不缓存 N 套状态 |
| E | 精确谱输出分配、共享/常量系数、场 kernel 优化 | 字节清单核实收益；一分量输出不再分配三分量谱；硬件计数器区分带宽、寻址/指令与占用问题；保持 Yee/CPML/边界公式及误差门槛 |
| F | 准确准入、B/K 调优，再做 Q=2 和多 GPU | 计入 prepare 过渡、Graph 池、输出、tape；多个 prepared 不超卖预算；OOM/取消/事件/失败传播测试；调优选择有测量依据 |
| G | Batch PyTorch 原生伴随 | B=1 与单场景梯度，B>1 与逐项梯度聚合对照；有限差分、共享参数、检查点与 tape 生命周期通过后才接受 requires_grad |

最终组合性能门继续沿用主计划：在相同输入、输出和精度条件下，完整 makespan 不超过十进程基线的 1.05 倍，同时满足显存目标；至少 2 次预热、5 次正式交错测量，报告中位数、范围及原始记录。需要覆盖三类负载、N=1/10/100 和原业务场景。若显存下降但吞吐不通过，应明确将其定位为显存受限时的选择。

## 8. 重跑入口

```bash
conda activate maxwell
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export CUDA_VISIBLE_DEVICES=1
export MAX_JOBS=4

python -m benchmark batch --profile launch --concurrency 10 --steps 128 --warmups 2 --repeats 5
python -m benchmark batch --profile local_spectrum --concurrency 10 --steps 64 --warmups 2 --repeats 5
python -m benchmark batch --profile bandwidth --concurrency 4 --steps 64 --warmups 2 --repeats 5
python -m benchmark batch --capacity-audit launch --variants 10 --capacities 1 2 4 10 --steps 128 --repeats 3

WITWIN_RUN_CUDA_EXTENSION_BUILD=1 python -m pytest \
  tests/api/public/test_batch_audit_regressions.py \
  tests/api/public/test_batch_execution.py \
  tests/api/public/test_batch_simulation.py \
  tests/fdtd/test_batch_audit_numerics.py \
  tests/fdtd/cuda/test_batch_runtime.py \
  tests/fdtd/cuda/test_cuda_batch.py \
  tests/boundaries/cpml/test_fdtd_cpml.py \
  tests/gradients/test_fdtd_adjoint_bridge.py \
  tests/validation/benchmark/test_batch_benchmark.py -q
```

profiling 默认保存到 `benchmark/plots/batch/`，也可用 `--output` 指定本报告中的路径。三个 GPU profiling 命令应依次运行。诊断用短步数不满足频谱物理收敛要求，不应用作天线或器件的业务精度验收。
