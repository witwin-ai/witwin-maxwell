from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed import DistributedFDTD
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig
from witwin.maxwell.result import Result
from witwin.maxwell.scene import prepare_scene


_FREQUENCY = 1.0e9
_TIME_STEPS = 48
_MONITOR_RTOL = 5.0e-5
_MONITOR_ATOL = 5.0e-6
_FLUX_REL_TOL = 1.0e-3


def _monitor_scene() -> mw.Scene:
    x = np.linspace(-0.5, 0.5, 11, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((float(x[0]), float(x[-1])), (float(y[0]), float(y[-1])), (float(z[0]), float(z[-1])))
        ),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.2, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=1.0),
            name="source",
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "y_plane",
            axis="y",
            position=0.05,
            fields=("Ez",),
            frequencies=(_FREQUENCY,),
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            "z_flux",
            axis="z",
            position=0.1,
            frequencies=(_FREQUENCY,),
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "x_plane",
            axis="x",
            position=0.1,
            fields=("Ez",),
            frequencies=(_FREQUENCY,),
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            "x_flux",
            axis="x",
            position=0.0,
            frequencies=(_FREQUENCY,),
        )
    )
    scene.add_monitor(
        mw.ModeMonitor(
            "z_mode",
            position=(0.0, 0.0, 0.1),
            size=(0.8, 0.3, 0.0),
            mode_index=0,
            polarization="Ex",
            frequencies=(_FREQUENCY,),
        )
    )
    return scene


def _solve(solver):
    return solver.solve(
        time_steps=_TIME_STEPS,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=False,
        normalize_source=False,
        use_cuda_graph=False,
    )


@pytest.fixture(scope="module")
def monitor_pair(cuda_p2p_devices):
    for device in cuda_p2p_devices:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    scene = _monitor_scene()
    single_solver = FDTD(prepare_scene(scene), frequency=_FREQUENCY, absorber_type="cpml")
    single_solver.init_field()
    single_output = _solve(single_solver)

    distributed_solver = DistributedFDTD(
        scene,
        frequency=_FREQUENCY,
        parallel=FDTDParallelConfig(
            devices=cuda_p2p_devices,
            transport="cuda_p2p",
            overlap=True,
            gather_fields=False,
            result_device=cuda_p2p_devices[0],
        ),
        absorber_type="cpml",
    )
    distributed_solver.init_field()
    distributed_output = _solve(distributed_solver)

    single_result = Result(
        method="fdtd",
        scene=scene,
        prepared_scene=single_solver.scene,
        frequency=_FREQUENCY,
        solver=single_solver,
        monitors=single_output["observers"],
        raw_output=single_output,
    )
    distributed_result = Result(
        method="fdtd",
        scene=scene,
        prepared_scene=distributed_solver.scene,
        frequency=_FREQUENCY,
        solver=distributed_solver,
        monitors=distributed_output["observers"],
        raw_output=distributed_output,
    )
    yield SimpleNamespace(
        single=single_output["observers"],
        distributed=distributed_output["observers"],
        single_result=single_result,
        distributed_result=distributed_result,
    )

    for device in cuda_p2p_devices:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _assert_coords_close(actual, expected) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-15,
    )


def _assert_tensor_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.to(expected.device),
        expected,
        rtol=_MONITOR_RTOL,
        atol=_MONITOR_ATOL,
    )


def _assert_raw_fields_close(actual, expected, fields) -> None:
    for component in fields:
        actual_component = actual["components"][component]
        expected_component = expected["components"][component]
        _assert_tensor_close(actual_component["data"], expected_component["data"])
        _assert_coords_close(actual_component["coords"][0], expected_component["coords"][0])
        _assert_coords_close(actual_component["coords"][1], expected_component["coords"][1])
        _assert_tensor_close(actual[component], expected[component])


def _assert_flux_close(actual, expected) -> None:
    for key in ("flux", "power"):
        actual_value = actual[key].to(expected[key].device)
        expected_value = expected[key]
        denominator = max(float(torch.abs(expected_value).item()), 1.0e-30)
        relative_error = float(torch.abs(actual_value - expected_value).item()) / denominator
        assert denominator > 1.0e-12
        assert relative_error <= _FLUX_REL_TOL


def test_y_normal_single_component_plane_monitor_matches_one_gpu(monitor_pair):
    expected = monitor_pair.single["y_plane"]
    actual = monitor_pair.distributed["y_plane"]
    _assert_raw_fields_close(actual, expected, ("Ez",))
    _assert_coords_close(actual["coords"][0], expected["coords"][0])
    _assert_coords_close(actual["coords"][1], expected["coords"][1])
    assert float(torch.abs(expected["Ez"]).max().item()) > _MONITOR_ATOL


def test_z_normal_flux_monitor_matches_fields_and_integrated_flux(monitor_pair):
    expected = monitor_pair.single["z_flux"]
    actual = monitor_pair.distributed["z_flux"]
    _assert_raw_fields_close(actual, expected, ("Ex", "Ey", "Hx", "Hy"))
    _assert_flux_close(actual, expected)


def test_off_split_x_normal_plane_monitor_matches_one_gpu(monitor_pair):
    expected = monitor_pair.single["x_plane"]
    actual = monitor_pair.distributed["x_plane"]
    _assert_raw_fields_close(actual, expected, ("Ez",))
    assert float(torch.abs(expected["Ez"]).max().item()) > _MONITOR_ATOL


def test_split_x_normal_flux_monitor_matches_fields_and_integrated_flux(monitor_pair):
    expected = monitor_pair.single["x_flux"]
    actual = monitor_pair.distributed["x_flux"]
    _assert_raw_fields_close(actual, expected, ("Ey", "Ez", "Hy", "Hz"))
    _assert_flux_close(actual, expected)


def test_mode_monitor_matches_raw_power_flux_and_resolved_mode(monitor_pair):
    expected_raw = monitor_pair.single["z_mode"]
    actual_raw = monitor_pair.distributed["z_mode"]
    _assert_raw_fields_close(actual_raw, expected_raw, ("Ex", "Ey", "Hx", "Hy"))
    _assert_flux_close(actual_raw, expected_raw)

    expected = monitor_pair.single_result.monitor("z_mode")
    actual = monitor_pair.distributed_result.monitor("z_mode")
    assert actual["effective_index"] == pytest.approx(expected["effective_index"], rel=1.0e-9)
    assert actual["beta"] == pytest.approx(expected["beta"], rel=1.0e-9)
    for key in (
        "mode_power",
        "total_power",
        "amplitude_forward",
        "amplitude_backward",
        "power_forward",
        "power_backward",
    ):
        _assert_tensor_close(actual[key], expected[key])
