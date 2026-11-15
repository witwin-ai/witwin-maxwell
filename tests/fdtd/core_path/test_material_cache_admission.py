"""FDTDMaterialCache runs on the core path: cached runs equal fresh runs bitwise."""
from dataclasses import replace

import pytest
import torch

import witwin.maxwell as mw
from benchmark.core_path.gates import make_simulation
from benchmark.core_path.scenes import CACHE_SCENES, GATE_SCENES, build_core_case

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="The core path requires CUDA.")


@pytest.mark.parametrize("name", CACHE_SCENES)
def test_cache_gate_scenes(name):
    assert name in GATE_SCENES
    case = build_core_case(name)
    assert case.name == CACHE_SCENES[name]
    assert case.material_cache == name.split("_")[1]
    assert (case.cache_warmup is not None) is (case.material_cache == "local")


def _fields(prepared):
    prepared.run()
    solver = prepared.solver
    assert solver._core_runtime is not None
    return {name: getattr(solver, name).clone() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}


@cuda
@pytest.mark.parametrize("name", ["cache_hit_mesh_dielectric", "cache_local_dielectric_slab"])
def test_cached_core_run_equals_fresh_core_run(name):
    case = build_core_case(name)
    fresh = _fields(make_simulation(case, steps=64, full_field_dft=False).prepare())
    cache = mw.fdtd.FDTDMaterialCache()
    warmup = case.scene if case.cache_warmup is None else case.cache_warmup
    outcomes = []
    for scene in (warmup, case.scene):
        prepared = make_simulation(replace(case, scene=scene), steps=64, full_field_dft=False,
                                   material_cache=cache).prepare()
        outcomes.append(prepared.solver._material_cache_info["outcome"])
        assert prepared.solver._core_runtime is not None
    assert outcomes == ["miss", case.material_cache]
    cached = _fields(prepared)
    assert prepared.solver._core_step_graph_active
    for component in fresh:
        assert torch.equal(fresh[component], cached[component]), component
