"""Genuine Cu short-horizon Graph and disk-resume regressions.

Controlled initial fields exercise the discrete state functional. These tests
make no continuum propagation, harmonic heat or attenuation claim.
"""
import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.copper_foil import compile_copper_faces
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene

FREQUENCIES = (0.8e9, 1e9, 1.2e9)


def _scene():
    nodes = np.arange(-5, 6, dtype=np.float64) * 0.0025
    scene = mw.Scene(
        domain=mw.Domain(bounds=tuple((float(nodes[0]), float(nodes[-1])) for _ in range(3))),
        grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=3),
        device="cpu",
        structures=[
            mw.Structure(
                mw.Box(
                    position=torch.zeros(3, dtype=torch.float64),
                    size=torch.tensor((0.01, 0.01, 0.0), dtype=torch.float64),
                ),
                mw.CopperFoil(thickness=35e-6, frequency_range=(0.8e9, 1.2e9)),
                name="film",
            )
        ],
    )
    prepared = prepare_scene(scene)
    face = next(face for face in compile_copper_faces(prepared, device="cpu") if face.component == 1)
    index = tuple(face.indices[len(face.indices) // 2].tolist())
    point = tuple(float(axis[i]) for axis, i in zip(component_coords(prepared, "Ey"), index))
    monitors = [
        mw.PlaneMonitor("published_Ey", axis="z", position=0.0, fields=("Ey",), frequencies=FREQUENCIES),
        mw.PointMonitor("same_owned_Ey", point, fields=("Ey",)),
    ]
    return scene.clone(monitors=monitors), index


def _run_control(graph, steps, directory, *, remaining=None):
    if not torch.cuda.is_available():
        pytest.skip("Actual CUDA execution required")
    scene, index = _scene()
    simulation = mw.Simulation.fdtd(
        scene.clone(device="cuda"),
        frequencies=FREQUENCIES,
        # The short-tail graph captures the GPU full-field DFT tail; request it explicitly
        # (before 6fc1d88c an explicit False was silently overridden to True).
        full_field_dft=True,
        field_dtype=torch.float32,
        cuda_graph=graph,
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
    )
    execution = simulation.prepare()
    solver = execution.solver
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        getattr(solver, name).zero_()
    solver.Ey.fill_(1.0)
    solver.Hx.fill_(0.25)
    for entry in solver._copper_foils:
        assert bool((entry["capacity"] > 0).all())
        if entry["electric_name"] == "Ey":
            entry["right_e"].fill_(3.0)

    if remaining is None:
        result = execution.run()
    else:
        checkpoint = execution.run_until(steps - remaining)
        assert checkpoint.step == steps - remaining and checkpoint.total_steps == steps
        checkpoint.save(directory / "checkpoint.pt")
        del checkpoint, execution
        restored = mw.FDTDResumeCheckpoint.load(directory / "checkpoint.pt", map_location="cpu")
        execution = simulation.prepare()
        result = execution.run(resume_from=restored)

    solver = result.solver
    assert solver._cuda_graph_active == graph
    assert solver._observer_cuda_graph_active == graph
    assert solver._tail_graph_active == graph, "Short tail capture cannot silently fall back to eager"
    assert result.stats()["steps_run"] == result.stats()["time_steps"] == steps
    assert all(count == steps for count in result.stats()["observer_sample_counts"])
    assert int(solver._dft_step.item()) == steps

    point = result.monitor("same_owned_Ey")["components"]["Ey"].detach().cpu()
    plane = result.monitor("published_Ey")["components"]["Ey"]["data"].detach().cpu()
    assert point.dtype == plane.dtype == torch.complex64
    torch.testing.assert_close(point.reshape(-1), plane[:, index[0], index[1]], rtol=0, atol=0)
    traces = result.conductor_losses["film"]["bulk_electric_traces"]
    assert len(traces) == 2 and {trace["component"] for trace in traces} == {"Ex", "Ey"}
    ey = next(trace for trace in traces if trace["component"] == "Ey")
    locations = torch.nonzero((ey["indices"].cpu() == torch.tensor(index)).all(dim=1)).flatten()
    assert locations.numel() == 1
    side = ey["electric_fields"].cpu()[:, :, int(locations[0])]
    scale = max(float(side.abs().max()), float(point.abs().max()))
    assert scale > 0
    assert float((side[:, 1] - point.reshape(-1).to(torch.complex128)).abs().max()) / scale > 1e-5

    state = {name: getattr(solver, name).detach().cpu().clone() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    for number, entry in enumerate(solver._copper_foils):
        for name in (
            "state", "right_e", "old_e", "cumulative_loss", "spectral_norm",
            "spectral_real", "spectral_imag", "spectral_step",
        ):
            state[f"Cu{number}_{name}"] = entry[name].detach().cpu().clone()
        torch.testing.assert_close(entry["spectral_norm"].cpu(), torch.full((3,), float(steps), dtype=torch.float64), rtol=0, atol=0)
    state["published_point"] = point.clone()
    state["published_plane"] = plane.clone()
    for number, trace in enumerate(traces):
        state[f"actual_Et_heads_{number}"] = trace["electric_fields"].detach().cpu().clone()

    result.save(directory / "result.pt")
    loaded = mw.Result.load(directory / "result.pt", scene=scene, map_location="cpu")
    torch.testing.assert_close(loaded.monitor("same_owned_Ey")["components"]["Ey"], point, rtol=0, atol=0)
    torch.testing.assert_close(loaded.monitor("published_Ey")["components"]["Ey"]["data"], plane, rtol=0, atol=0)
    restored_traces = loaded.conductor_losses["film"]["bulk_electric_traces"]
    assert len(restored_traces) == len(traces)
    assert tuple(trace["component"] for trace in restored_traces) == tuple(trace["component"] for trace in traces)
    for original, restored_trace in zip(traces, restored_traces):
        torch.testing.assert_close(restored_trace["electric_fields"], original["electric_fields"], rtol=0, atol=0)
    return state


@pytest.fixture(scope="module")
def direct_control(tmp_path_factory):
    cache = {}

    def get(graph, steps):
        key = graph, steps
        if key not in cache:
            directory = tmp_path_factory.mktemp(f"short-direct-N{steps}-graph{int(graph)}")
            cache[key] = _run_control(graph, steps, directory)
        return cache[key]

    return get


@pytest.mark.parametrize("steps", [1, 2, 3, 4])
@pytest.mark.parametrize("graph", [False, True])
def test_short_tail_captures_and_preserves_actual_Cu_state(steps, graph, direct_control):
    actual = direct_control(graph, steps)
    reference = direct_control(False, steps)
    assert actual.keys() == reference.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)


@pytest.mark.parametrize("remaining", [1, 2, 3])
@pytest.mark.parametrize("graph", [False, True])
def test_last_few_steps_resume_matches_complete_Cu_run(remaining, graph, direct_control, tmp_path):
    reference = direct_control(graph, 4)
    resumed = _run_control(graph, 4, tmp_path, remaining=remaining)
    assert resumed.keys() == reference.keys()
    for name in resumed:
        torch.testing.assert_close(resumed[name], reference[name], rtol=0, atol=0)
