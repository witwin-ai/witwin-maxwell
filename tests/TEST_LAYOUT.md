# 可扩展测试分层（当前状态）

本次按功能将 tests 重构为“顶层域 -> 子功能 -> 用例文件”。

- `api/`
  - `public/`: 公开 API 与运行入口
  - `adapters/tidy3d/`: Tidy3D 适配器
  - `adapters/gds/`: GDS 适配器
- `boundaries/`
  - `boundary_specs/`: 边界类型映射与状态初始化
  - `boundary_physics/`: 边界物理行为（周期、PEC、Bloch、PMC 等）
  - `cpml/`: PML/CPML 配置与吸收行为
- `core/`
  - `scene/`: Scene 组装与几何行为
  - `geometry/`: 几何/网格资产与体素化
  - `grid/`: 坐标与网格工具
- `materials/`
  - `compiler/`: 材料编译与体素重建
  - `dispersive/`: 色散介质行为
- `monitors/`
  - `observers/`: 监视器采样、重建与后处理桥接
- `sources/`
  - `definitions/`: 源对象及编译行为
  - `point/`: 点源与源归一化
  - `incident/`: 入射场辅助网格
  - `tfsf/`: TFSF 注入
- `postprocess/`
  - `directivity/`: 方向图/增益指标
  - `scattering/`: S-参数
  - `rcs/`: RCS 与近/远场转换
  - `evaluation/`: 结果误差与对齐工具
- `validation/`
  - `benchmark/`: benchmark 流水线与对比工具
  - `cross_solver/`: FDFD 与 FDTD 交叉验证
- `gradients/e2e/`: 预留：梯度端到端检测

未来添加新场景/新功能时，请沿对应顶层域新增子文件夹（如 `gradients/e2e/` 下新增 `test_*`）。
