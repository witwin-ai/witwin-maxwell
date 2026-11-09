# PCB FDTD 功能与验收矩阵

状态：实施中。R1、R2 均未发布；本表区分已实现的离散合同和完整物理验收。
维护测试的完整冻结源回归正在执行、尚未结束，不能把多个阶段的测试数量相加当作最终通过。
逐阶段提交、原始证据和剩余门槛见[实施台账](../assessments/pcb-development-2026-10-22/implementation-ledger.md)，
验收范围见[完整计划](../plans/fdtd-pcb-antenna-development-and-test-plan-2026-10-22.md)。

| 几何与材料 | 馈电和离散合同 | 已有执行/观察证据 | 物理验收状态 |
| --- | --- | --- | --- |
| Cartesian 体积 PEC、平面 PEC 多边形/孔槽 | 原生切向边，显式端子、有限电压孔径与实际磁环 | 场、连接/gap、Graph 和恢复分项控制 | 基础合同 passed；完整天线及外部参考仍 unconverged |
| 静态有损介质与导电薄片 | 原生 Yee σ，与端口、RLC/MNA/网络联合更新 | 介质功率、源功、被动性、参数梯度及运行状态控制 | 小问题 passed；新完整组合需冻结回归 |
| 介质 ADE＋轴对齐铜箔 | 双侧实际场和铜扩散状态联合消元，物理厚度和频带显式 | 平膜参考、长期稳定性、Graph/恢复及连续材料梯度 | 分项 passed；完整 PCB 边缘/模型收敛未完成 |
| 微带、stripline、CPW、差分对 | 真实多驱动 V/I、功率对偶共/差模；模式观察单独核算 | 五频点物理转移、部分独立全波参考、区域热/通量 | β/Z 有局部 passed；连续体、全频段 α/modal-S 整体验收开放 |
| patch 与印刷偶极子 | 有限馈电孔径，固定物理观察面及外部区域 | 全频段方向图、损耗、净流、NF2FF 和分离控制 | 能量与部分时间/边界 passed；组合空间及独立参考 unconverged |
| 五类应用候选 | 显式导体选择、真实损耗与 direct/import 离散身份 | 12 ns 单频路径/功率，材料法向积分及独立边界/空间控制 | 单频部分 passed；完整声明频段与组合收敛 not_run/unconverged |
| 固定正交 `BendCoordinates` | 真实层间距、度量、chart-normal 双侧铜与 RF 馈电 | world 查询/NF2FF、物理材料平均、Graph/恢复；独立弯曲参考 | 受限实现 passed；完整 R2 unconverged。独立弯曲参考：修正版 625 µm 细网格对 1250 µm 复数远场 4.083% 未过原 2% 门（保持 FAIL）；2026-10-29 用户决定采用 fine625 为弯曲参考，误差带复数 4.7%／功率 5.7%（Richardson p=1 加外部项），Maxwell 对照容差须 ≥ 自身门＋该误差带并注明（[验收](../assessments/bend-graded-angular-corrected-fine-field-launch-2026-10-29/ACCEPTANCE.md)） |
| 平面任意斜置 `SurfaceMesh`＋有限铜/PEC（`sheet_model="conformal_two_sided"`） | 共形双侧 cut complex；局部隐式 Newmark（beta=1/4），显式 leapfrog 界与 PEC 一致。D6 支持电阻性 `LumpedPort` sheet gap feed、原生 V/I 与含 feed 的 checkpoint/resume；馈电轴必须 Cartesian 且位于支撑面内，非 Cartesian 方向显式拒绝 | 原生 CUDA/CPU 私有 float64 单步差 <=2.63e-16，graph/eager 位级一致；原有平面散射门 passed。F4 modified energy 最大/初始=1，最终/初始=0.12151；F2 两种倾角 λ/40 Z_in 差 212.77%/185.23%，λ/60 为 168.03%/165.82%，共振偏移 18.12–20.16%，FAIL；D_max 差 <0.054 dB | F2 天线旋转不变性未验收，不能由散射门推导。F3 λ/60 最大残差：倾斜偶极子 0.02849%、倾斜 patch 0.001689%，通过 1% 门。F2/F2b/F3 逐项结果见 [D6 assessment](../assessments/thin-sheet-feeds-validation-2026-11.md)。V15 性能门仍未验收；周期边界、非 gap 端口、反应性负载/电路/网络、面上独立源、梯度、批处理、分片及无 feed 的恢复显式拒绝 |
| 分片平面（折面）`SurfaceMesh`＋PEC/有限铜（弯折板、闭合棱柱/多面柱） | 折面弦迹沿折点的两段路径积分，闭合 cut-cell 环提供每侧独立法向磁通；折楔精确裁剪对偶体积。D6 gap 和沿馈电轴的一格余量必须在同一平面 facet；臂上的远处折面允许，Cartesian 面内 feed 约束不变 | 原有 32 面柱 λ/40 散射误差 TE 0.343%/TM 0.425%，铜-PEC 差 3.61e-5/6.27e-5，通过 V13。D6 λ/40、λ/60、λ/80 自收敛 FAIL：共振/高频边缘的 Z_in 增量为 178.6847/326.0230 Ω → 178.9352/395.6786 Ω；λ/60→λ/80 D_max 差 0.01641 dB passed。λ/60 F3 最大残差 0.015023%，通过 1% 门，详见 [D6 assessment](../assessments/thin-sheet-feeds-validation-2026-11.md) | 原散射门 passed；D6 按独立天线门记录，不与平面偶极子作物理等价比较。小于约一个网格的 facet、>90° 折角、三面交点及 gap/余量内折面仍显式拒绝 |
| 1/2 GPU 空间分片 | 唯一边/铜/ADE/端口 owner 与观察 halo | 实际有限铜天线场/损耗/远场、磁盘恢复的双卡对照 | 正确性分项 passed；规模性能与未实现连接组合未验收 |

“passed”仅指单元格声明的具体控制。表中没有宣称任意两项的笛卡尔积都可组合。
完整执行模式必须用实际测试覆盖，不能把 eager 成功直接推广为 Batch、空间分片或梯度成功。

## 铜界面观察与功率

| 操作 | 当前边界 |
| --- | --- |
| `Result.query_fields` 默认六字段单侧铜查询 | 保留单频 E/H tuple、真实左右 Et 与同侧 Ht/En 重建及原共同谱尺度合同；需要完整 full-field/Point 原生支撑，映射路线保持原范围 |
| `Result.query_fields(components=..., frequencies=...)` | 返回 dict 中 `[频率, 点]`；真实 Cu owner/side、有限 complex128 两侧头和实际 per-face clock/norm；`published_native` 仅负侧原生 primitive，允许 float64 权重提升算术精度；不从 left/native 拟合相位或尺度 |
| selected 查询初期运行范围 | 真实 `real_projected` 时间 bank、Cartesian 非周期、固定 float64 位置、total、raw source normalization、none 窗；只用实际 native Point/单法向 plane，派生 plane 不遮挡完整 Point；Graph/eager、磁盘恢复、保存/公开重载和场/头及铜材料 VJP 已有有界实际控制 |
| selected 尚未实现的扩展 | 坐标/shape 梯度、analytic complex bank、mapped、periodic/Bloch、source-normalization 撤销、weighted-window 及 matched-reference incident/scattered 均明确 NI；缺 provenance 的旧结果不能伪补为新 strict 查询通过 |
| Cartesian 原生 plane/closed 的配对通量 | 使用真实双侧 Et、固定单位磁导率及物理面积；片内切向穿越、同侧 H 支撑的片外法向插值已通过功能/梯度控制；精确落片、跨片 H 模板、缺失谱头及不完整旧归一化来源仍拒绝 |
| 普通 point 与其他跨铜插值 | 需要其自身完整单侧支撑；不能用已验证的配对通量或查询能力替未验证模板背书 |
| 实场 Cartesian 时域通量 | 原生配对乘积、真实铜右迹与预分配 GPU 缓冲；明确输出 E(t)/H(t−dt/2)，不提供时域梯度；mapped、磁性、法向落片与端点接缝仍受能力门限制 |
| Cartesian 有限区域静态损耗 | 共享 `physical_intersection` 体积、铜半体积与 tile 功率；周期/映射新 ROI 范围仍受限 |
| 弯曲观察 | 受限固定坐标中的 world 位置、共因子面积、向量和相位已实现；不代表一般 cut-surface 铜已实现 |

完整区域损耗、瞬时算法能量和频域周期平均热量是不同量。复数双 bank 的算法能量
不能直接当成真实瞬态焦耳热；物理功率必须使用对应的谱幅度约定和物理入射功率。
复数 CW 的绝对幅相修复及其有限窗、普通 forward FD、Graph/磁盘恢复控制已完成分项验证，
见[阶段报告](../assessments/fdtd-analytic-cw-source-contract-2026-10-23.md)。完整维护套件与发布门槛仍单独验收。

配对通量修复在同一源码、真实 core 0.4.1 与精确 ABI4 库上通过 39 项实际 GPU
控制（15 频域、18 时域、6 既有时间观察器），零跳过。主仓合入 CPU 子集为
625 passed、23 个既有 CUDA skip、16 个明确排除的 CUDA 用例；这些 skip 不计作通过。
新 14 项能力边界中，13 项有精确 raise 事件，1 项采样器结构不变量保持待覆盖。
完整维护套件、两长度衰减与连续体误差仍须单独验收。

## 连续梯度与离散事件

固定网格、拓扑及所有权下，已有实际普通 forward 中心差分覆盖介质参数、静态 σ、
铜厚/电导、局部 ADE 和支持的电路参数。结果查询对保留场的可微后处理，不能代替
从场求解器到目标的端到端伴随验收。新复数源表示、弯曲度量、Batch 和空间分片的
梯度各自按明确覆盖声明；未注册组合继续拒绝。

无铜的 `PowerLossMonitor` 直接 σ 输出头及 native batch 所有权缺口已修复，
提交 `b27570b` 的四项公共 GPU 控制覆盖普通 forward 差分、Graph、磁盘恢复、
重复 batch 与关闭后的反传；最大差分误差为 0.08755%。旧的 58 项 G11 节点
没有覆盖该组合，不能用于追认旧结果。

固定弯曲材料现有受限公共准入：固定 chart／网格／拓扑、显式 dt 与步数、单频、
单设备、实场、固定 Norton 馈电和观察，允许注册的标量 ε、σ 与固定阶数铜厚／电导。
不支持磁性、ADE、PEC 组合、几何／源／电路／chart 导数或 mapped native batch。
真实公共 24 项 GPU 矩阵通过原四步长／1% 规则；仅弱负曲率辐射功率的普通差分
参考显式采用 FP64，伴随主场仍为 FP32。Graph／恢复控制和 Cartesian native batch
是各自声明的测试范围。主仓 475 CPU 与 7 项能力清单检查通过。

首轮公共 3 passed / 21 failed 的 no_grad 依赖检查问题及此前私有 23/24 失败均保留。
修复后通过不追认历史结果，也不替代最终冻结源完整回归、完整 G11 或 R2 验收。

`field_dtype=torch.float64` 与显式 FP64 稀疏观察已在受限、detached、固定网格、
单设备、单频、eager、全六面 dense CPML 的公共前向路径中实现。
268 CPU、3 项公共 GPU 加 4 项 core GPU、52 项原默认精度 CUDA 对照通过；
不代表 FP64 伴随、Graph、Batch 或空间分片能力。

该既有路径的 dense CPML 电／磁导数、卷积记忆与 curl 中间量现保持字段 scalar type，
修复 double 缓冲区内中间值窄化到 float 的精度缺陷。原库 36 组完整网格控制的失败保留；
新库通过原 `2e-10` 更新门限，72 组默认精度输出与原库逐位相同。
同一主仓源码的新增 6 项 CPU 参考、12 项实际六分量 CPML GPU 与 3 项原公共 GPU
组成 21 项实际控制，零跳过；其 96 步公开样本属于精度／持久化合同，不替代完整维护回归或天线收敛。
详见[算术范围](fdtd-native-fp64-cpml.md)，既有执行／材料准入与曲面铜边界不变。

复数 CW/Bloch 铜在无 bulk conduction 的背景中已有端到端分项梯度控制；这不代表
复数 bulk σ 的反向更新已实现。复数导电体背景、mapped 端到端伴随和空间分片伴随
仍按现有能力门拒绝。CW 幅度/相位仍为公共浮点控制，并未新增波形 Tensor 参数梯度。

插面、导体开关、连接变化和 hard PEC 边所有权变化均为离散事件。没有把它们转换
为平滑几何梯度，也没有以零梯度或 detach 来伪装支持。

## 仍须完成的发布工作

R1 仍需传输线连续体/衰减/modal-S、天线组合收敛与独立参考、五类应用完整频段、
同误差真实体积铜性能比较、最终维护回归及统一 benchmark 报告。R2 另外需要
完整复杂几何、外部弯曲天线、声明的连续梯度/跨设备组合和规模性能证据。
失败、未运行、未收敛和明确不支持均按原状态保留，不折算为通过。


最新全时长CPW／差分线功能矩阵为12／12列通过，完整相干功率闭合最大约0.01731%。
两个真实DUT长度、完整独立驱动及固定物理观察均有原始数据审计；局部离散传播诊断仍不关闭
连续体铜损、模式S或网格收敛门。见[差分线完整审计](../assessments/printed-lines-2026-10-28/coupled-full64-current740/pair-eight-complete/README.md)。
真正三轴研究推进器另完成四个4096步场控制及保存状态的独立守恒复核，见[实际阶段](../assessments/genuine-three-dimensional-stepper-v2-actual-2026-10-28/README.md)。
该研究推进器没有生产Scene背景集成；表中的曲面有限铜仍为unsupported；平面斜置薄片的原生 CUDA 与 CPU 参考运行时见上表。


Selected-query 扩展的 [CPU/能力边界准备证据](../assessments/printed-lines-2026-10-28/public-selected-query-v3/preparation/README.md) 与 [实际 11 项 GPU/独立 raw 审计](../assessments/printed-lines-2026-10-28/public-selected-query-v3/actual/README.md) 分开保留。108 CPU、7 exact-census 和实际 11 GPU 不是整个维护套件；445 项能力欠账与29项原合同排除也不是功能通过计数。默认 single-frequency tuple 路线与新 selected 路线范围不同，不能把新接口的 fixed-position VJP 推广成坐标/shape 导数或任意组合资格。
