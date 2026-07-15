from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed import DistributedFDTD
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig
from witwin.maxwell.scene import prepare_scene


_FREQUENCY = 1.0e9
_TIME_STEPS = 24
_FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@dataclass
class _SolvePair:
    single_solver: FDTD
    single_output: dict
    distributed_solver: DistributedFDTD
    distributed_output: dict


def _grid_nodes(*, nonuniform_x: bool):
    if nonuniform_x:
        # Eleven uneven x cells give a 6/5 partition. The finer, replicated y/z
        # spacing keeps the shard-local CFL time steps exactly equal.
        x = np.asarray(
            [-0.55, -0.44, -0.33, -0.21, -0.10, 0.00, 0.09, 0.19, 0.30, 0.42, 0.53, 0.65],
            dtype=np.float64,
        )
    else:
        x = np.linspace(-0.5, 0.5, 11, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    return x, y, z


def _vacuum_scene(*, nonuniform_x: bool, source_on_interface: bool) -> mw.Scene:
    x, y, z = _grid_nodes(nonuniform_x=nonuniform_x)
    interface_node = (len(x) - 1 + 1) // 2
    source_x = float(x[interface_node] if source_on_interface else x[2])
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (float(x[0]), float(x[-1])),
                (float(y[0]), float(y[-1])),
                (float(z[0]), float(z[-1])),
            )
        ),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )
    scene.add_source(
        mw.PointDipole(
            position=(source_x, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(
                frequency=_FREQUENCY,
                amplitude=1.0,
            ),
            name="interface_source" if source_on_interface else "source",
        )
    )
    scene.add_monitor(
        mw.PointMonitor(
            "probe",
            (source_x, 0.0, 0.0),
            fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        )
    )
    return scene


def _run_single(scene: mw.Scene):
    solver = FDTD(prepare_scene(scene), frequency=_FREQUENCY, absorber_type="cpml")
    solver.init_field()
    output = solver.solve(
        time_steps=_TIME_STEPS,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=False,
        normalize_source=False,
        use_cuda_graph=False,
    )
    return solver, output


def _run_distributed(
    scene: mw.Scene,
    devices: tuple[torch.device, torch.device],
    *,
    gather_fields: bool,
):
    config = FDTDParallelConfig(
        devices=devices,
        transport="cuda_p2p",
        overlap=True,
        gather_fields=gather_fields,
        result_device=devices[0],
    )
    solver = DistributedFDTD(
        scene,
        frequency=_FREQUENCY,
        parallel=config,
        absorber_type="cpml",
    )
    solver.init_field()
    output = solver.solve(
        time_steps=_TIME_STEPS,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=False,
        normalize_source=False,
        use_cuda_graph=False,
    )
    return solver, output


def _assert_step_field_close(actual: torch.Tensor, reference: torch.Tensor) -> None:
    actual = actual.to(reference.device)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(reference).all()
    absolute = torch.abs(actual - reference)
    assert float(absolute.max().item()) <= 2.0e-6

    reference_scale = float(torch.abs(reference).max().item())
    significant = torch.abs(reference) >= max(1.0e-6, 1.0e-4 * reference_scale)
    if bool(significant.any()):
        relative = absolute[significant] / torch.abs(reference[significant])
        assert float(relative.max().item()) <= 2.0e-5


def _distributed_step_field(pair: _SolvePair, name: str) -> torch.Tensor:
    if name in pair.distributed_output:
        return pair.distributed_output[name]
    local_values = tuple(
        getattr(shard.solver, name) for shard in pair.distributed_solver.shards
    )
    return pair.distributed_solver._gather_component(name, local_values)


def _monitor_component(payload: dict, name: str) -> torch.Tensor:
    value = payload["components"][name]
    if isinstance(value, dict):
        value = value["data"]
    return torch.as_tensor(value)


@pytest.fixture(scope="module")
def uniform_vacuum_pair(cuda_p2p_devices):
    scene = _vacuum_scene(nonuniform_x=False, source_on_interface=False)
    single_solver, single_output = _run_single(scene)
    distributed_solver, distributed_output = _run_distributed(
        scene, cuda_p2p_devices, gather_fields=True
    )
    yield _SolvePair(single_solver, single_output, distributed_solver, distributed_output)
    del single_solver, single_output, distributed_solver, distributed_output
    for device in cuda_p2p_devices:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def uneven_interface_pair(cuda_p2p_devices):
    scene = _vacuum_scene(nonuniform_x=True, source_on_interface=True)
    single_solver, single_output = _run_single(scene)
    distributed_solver, distributed_output = _run_distributed(
        scene, cuda_p2p_devices, gather_fields=True
    )
    yield _SolvePair(single_solver, single_output, distributed_solver, distributed_output)
    del single_solver, single_output, distributed_solver, distributed_output
    for device in cuda_p2p_devices:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


@pytest.mark.parametrize("fixture_name", ["uniform_vacuum_pair", "uneven_interface_pair"])
def test_two_gpu_step_fields_match_one_gpu(request, fixture_name):
    pair: _SolvePair = request.getfixturevalue(fixture_name)
    for name in _FIELD_NAMES:
        _assert_step_field_close(
            _distributed_step_field(pair, name),
            getattr(pair.single_solver, name),
        )


@pytest.mark.parametrize("fixture_name", ["uniform_vacuum_pair", "uneven_interface_pair"])
def test_two_gpu_point_monitor_matches_one_gpu(request, fixture_name):
    pair: _SolvePair = request.getfixturevalue(fixture_name)
    single = pair.single_output["observers"]["probe"]
    distributed = pair.distributed_output["observers"]["probe"]
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(
            _monitor_component(distributed, name).to("cuda:0"),
            _monitor_component(single, name).to("cuda:0"),
            rtol=5.0e-5,
            atol=5.0e-6,
        )


def test_gather_false_returns_monitors_without_global_fields_and_reports_stats(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    scene = _vacuum_scene(nonuniform_x=False, source_on_interface=True)
    solver, output = _run_distributed(scene, cuda_p2p_devices, gather_fields=False)

    assert output is not None
    assert set(output) == {"observers"}
    assert "probe" in output["observers"]
    stats = solver.parallel_stats
    assert stats["devices"] == ("cuda:0", "cuda:1")
    assert stats["transport"] == "cuda_p2p"
    assert stats["gather_fields"] is False
    assert stats["result_device"] == "cuda:0"
    assert stats["overlap_requested"] is True
    assert stats["overlap_active"] is True
    assert stats["halo_bytes_per_step"] > 0
    assert stats["halo_bytes_total"] == stats["halo_bytes_per_step"] * _TIME_STEPS
    assert set(stats["peak_memory_bytes"]) == {"cuda:0", "cuda:1"}
    assert all(value > 0 for value in stats["peak_memory_bytes"].values())
    assert stats["topology"]["neighbor_pairs"][0]["peer_left_to_right"] is True
    assert stats["topology"]["neighbor_pairs"][0]["peer_right_to_left"] is True


def test_gather_true_places_owned_global_fields_on_requested_result_device(
    uniform_vacuum_pair,
):
    pair = uniform_vacuum_pair
    output = pair.distributed_output
    solver = pair.distributed_solver

    expected_shapes = {
        "Ex": (solver.Nx - 1, solver.Ny, solver.Nz),
        "Ey": (solver.Nx, solver.Ny - 1, solver.Nz),
        "Ez": (solver.Nx, solver.Ny, solver.Nz - 1),
    }
    for name, shape in expected_shapes.items():
        assert tuple(output[name].shape) == shape
        assert output[name].device == torch.device("cuda:0")
    assert solver.parallel_stats["gather_fields"] is True
    assert len(solver.parallel_stats["partitions"]) == 2
    assert solver.parallel_stats["partitions"][0]["physical_cells"][0] == 0
    assert (
        solver.parallel_stats["partitions"][-1]["physical_cells"][1]
        == solver.partition_plan.physical_cell_count
    )


def _feature_scene(case: str) -> mw.Scene:
    x = np.linspace(-0.6, 0.6, 13, dtype=np.float64)
    y = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
    z = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
    if case == "xyz_cpml_dielectric":
        boundary = mw.BoundarySpec.pml(num_layers=2, strength=1.0e6)
    elif case == "mixed_pec_pmc_mur":
        boundary = mw.BoundarySpec.faces(
            default="none",
            x=("pec", "pmc"),
            y=("mur", "mur"),
            z=("pec", "mur"),
        )
    else:
        boundary = mw.BoundarySpec.none()

    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (float(x[0]), float(x[-1])),
                (float(y[0]), float(y[-1])),
                (float(z[0]), float(z[-1])),
            )
        ),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=boundary,
        device="cuda:0",
    )
    if case == "xyz_cpml_dielectric":
        scene.add_structure(
            mw.Structure(
                name="interface_dielectric",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.4, 0.4)),
                material=mw.Material(eps_r=3.0),
            )
        )
    elif case == "conductive_electric_dispersion":
        scene.add_structure(
            mw.Structure(
                name="lossy_lorentz_interface",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.4, 0.4)),
                material=mw.Material(
                    eps_r=2.0,
                    sigma_e=0.02,
                    lorentz_poles=(
                        mw.LorentzPole(
                            delta_eps=0.5,
                            resonance_frequency=2.0e9,
                            gamma=2.0e8,
                        ),
                    ),
                ),
            )
        )

    if case == "uniform_current_interface":
        scene.add_source(
            mw.UniformCurrentSource(
                size=(0.5, 0.3, 0.3),
                center=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=mw.CW(
                    frequency=_FREQUENCY,
                    amplitude=1.0,
                ),
                name="cross_interface_current",
            )
        )
    else:
        scene.add_source(
            mw.PointDipole(
                position=(-0.3, 0.0, 0.0),
                polarization="Ez",
                profile="ideal",
                source_time=mw.CW(
                    frequency=_FREQUENCY,
                    amplitude=1.0,
                ),
                name="source",
            )
        )
    scene.add_monitor(
        mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=_FIELD_NAMES)
    )
    return scene


@pytest.mark.parametrize(
    "case",
    (
        "xyz_cpml_dielectric",
        "conductive_electric_dispersion",
        "mixed_pec_pmc_mur",
        "uniform_current_interface",
    ),
)
def test_two_gpu_feature_matrix_matches_one_gpu(
    case,
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    scene = _feature_scene(case)
    single_solver, single_output = _run_single(scene)
    distributed_solver, distributed_output = _run_distributed(
        scene,
        cuda_p2p_devices,
        gather_fields=True,
    )
    pair = _SolvePair(
        single_solver,
        single_output,
        distributed_solver,
        distributed_output,
    )

    for name in _FIELD_NAMES:
        _assert_step_field_close(
            _distributed_step_field(pair, name),
            getattr(single_solver, name),
        )
        torch.testing.assert_close(
            _monitor_component(distributed_output["observers"]["probe"], name).to("cuda:0"),
            _monitor_component(single_output["observers"]["probe"], name).to("cuda:0"),
            rtol=5.0e-5,
            atol=5.0e-6,
        )

    if case == "xyz_cpml_dielectric":
        assert all(shard.solver.uses_cpml for shard in distributed_solver.shards)
        for shard in distributed_solver.shards:
            assert shard.solver.scene.boundary.uses_kind("pml")
    if case == "uniform_current_interface":
        assert all(shard.solver._source_terms for shard in distributed_solver.shards)
