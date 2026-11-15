"""Bent coordinate charts on the core path: admission and whole-step graph runs."""
import pytest
import torch

from benchmark.core_path.gates import make_simulation
from benchmark.core_path.scenes import CHART_SCENES, GATE_SCENES, build_core_case

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="The core path requires CUDA.")


@pytest.mark.parametrize("name", CHART_SCENES)
def test_chart_gate_scenes_are_admitted(name):
    assert name in GATE_SCENES
    case = build_core_case(name)
    assert case.scene.coordinates is not None


@cuda
@pytest.mark.parametrize("name", CHART_SCENES)
def test_chart_scenes_run_whole_step_graphs(name):
    case = build_core_case(name)
    prepared = make_simulation(case, steps=64, full_field_dft=True).prepare()
    solver = prepared.solver
    assert solver._core_runtime is not None
    # The metric is folded into the constitutive tables once, at compile time.
    assert "coordinate_frame" in solver._compiled_material_model
    assert all("material_metric_factors" in entry for entry in solver._copper_foils)
    prepared.run()
    assert solver._core_runtime is not None
    assert solver._core_step_graph_active and solver._core_step_graph_k == 32
    assert all(bool(torch.isfinite(getattr(solver, name)).all()) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
