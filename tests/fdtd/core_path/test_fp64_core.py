"""Float64 fields on the core path: admission, the float64 fused Yee step and whole-step graph runs."""
import pytest
import torch

from benchmark.core_path.gates import make_simulation
from benchmark.core_path.scenes import FP64_SCENES, GATE_SCENES, build_core_case
from tests.core.scene.fp64_public_scene import scene as public_scene, simulation as public_simulation
from tests.support.numerical_equivalence import assert_numerically_equivalent

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="The float64 core path requires CUDA.")
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@pytest.mark.parametrize("name", FP64_SCENES)
def test_fp64_gate_scenes_are_admitted(name):
    assert name in GATE_SCENES
    case = build_core_case(name)
    assert case.field_dtype == torch.float64 and case.cpml_memory == "dense"


def _prepared(name):
    case = build_core_case(name)
    prepared = make_simulation(case, steps=64, full_field_dft=True).prepare()
    solver = prepared.solver
    assert solver._core_runtime is not None and solver.Ex.dtype == torch.float64
    return case, prepared, solver


def _state(solver):
    return {name: value for name, value in vars(solver).items()
            if (name in FIELDS or name.startswith("psi_")) and isinstance(value, torch.Tensor)}


@cuda
@pytest.mark.parametrize("name", ("fp64_vacuum_port", "fp64_slab_port"))
def test_fp64_fused_step_matches_the_yee_oracle(name):
    # Uniform (vacuum) and indexed (graded slab) coefficients, dense CPML: two fused H+E steps
    # against the Torch Yee oracle (the per-component kernels were deleted in W7 row 16e)
    # from the same random state.
    from tests.oracles import yee_cpu_reference
    _, _, solver = _prepared(name)
    state = _state(solver)
    generator = torch.Generator(device=solver.device).manual_seed(12)
    for tensor in state.values():
        tensor.copy_(torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype))
    start = {name: tensor.clone() for name, tensor in state.items()}
    for _ in range(2):
        solver._core_runtime.magnetic()
        solver._core_runtime.electric()
    core = {name: tensor.clone() for name, tensor in state.items()}
    for name, tensor in state.items():
        tensor.copy_(start[name])
    for _ in range(2):
        yee_cpu_reference.magnetic_step(solver)
        yee_cpu_reference.electric_step(solver)
    assert_numerically_equivalent(state, core, level="step")


@cuda
def test_fp64_whole_step_graph_is_deterministic():
    runs = []
    for _ in range(3):
        _, prepared, solver = _prepared("fp64_slab_port")
        result = prepared.run()
        assert solver._core_step_graph_active and solver._core_step_graph_k == 32
        assert solver._core_runtime is not None
        surface = result._native_raw_monitor("radiation")["faces"]
        runs.append([getattr(solver, name).clone() for name in FIELDS]
                    + [face[component] for face in surface.values() for component in face["fields"]])
    for run in runs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(runs[0], run))


@cuda
def test_fp64_graph_runs_match_eager_with_and_without_the_chart():
    # The public fixture runs on the core path with and without its bend chart.
    for curvature in (-3.0, None):
        value = public_scene(curvature=curvature)
        value = value.updated(monitors=tuple(m for m in value.monitors if type(m).__name__ != "PowerLossMonitor"))
        outputs = []
        for graph in (False, True):
            result = public_simulation(value, cuda_graph=graph).run()
            assert result.solver._core_runtime is not None
            assert getattr(result.solver, "_core_step_graph_active", False) is graph
            faces = result._native_raw_monitor("radiation")["faces"]
            outputs.append({f"{face}/{component}": data[component] for face, data in faces.items()
                            for component in data["fields"]}
                           | {name: getattr(result.solver, name) for name in FIELDS})
        assert_numerically_equivalent(outputs[0], outputs[1], level="run")
