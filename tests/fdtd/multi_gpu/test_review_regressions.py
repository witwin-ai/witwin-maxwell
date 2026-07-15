from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed import DistributedFDTD
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig
from witwin.maxwell.scene import prepare_scene


_FREQUENCY = 1.0e9
_FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _parallel(devices, *, gather_fields: bool) -> FDTDParallelConfig:
    return FDTDParallelConfig(
        devices=devices,
        transport="cuda_p2p",
        overlap=True,
        gather_fields=gather_fields,
        result_device=devices[0],
    )


def _direct_distributed(scene, devices, *, gather_fields: bool) -> DistributedFDTD:
    solver = DistributedFDTD(
        scene,
        frequency=_FREQUENCY,
        parallel=_parallel(devices, gather_fields=gather_fields),
        absorber_type="cpml",
    )
    solver.init_field()
    return solver


def _regular_scene() -> mw.Scene:
    x = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 5, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 5, dtype=np.float64)
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
            position=(-0.2, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=1.0),
            name="source",
        )
    )
    return scene


def test_nanometer_scale_monitors_on_interface_sides_have_unique_owner(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    x = np.linspace(-4.0e-9, 4.0e-9, 9, dtype=np.float64)
    y = np.linspace(-2.0e-9, 2.0e-9, 5, dtype=np.float64)
    z = np.linspace(-2.0e-9, 2.0e-9, 5, dtype=np.float64)
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
    scene.add_monitor(
        mw.PointMonitor("left_of_interface", (-0.25e-9, 0.0, 0.0), fields=("Ez",))
    )
    scene.add_monitor(
        mw.PointMonitor("right_of_interface", (0.25e-9, 0.0, 0.0), fields=("Ez",))
    )

    solver = _direct_distributed(scene, cuda_p2p_devices, gather_fields=False)
    owners = {monitor.name: [] for monitor in scene.monitors}
    for shard in solver.shards:
        for monitor in shard.solver.scene.monitors:
            owners[monitor.name].append(shard.rank)

    assert owners == {
        "left_of_interface": [0],
        "right_of_interface": [1],
    }


def test_monitor_output_keeps_scene_declaration_order_across_shards(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    scene = _regular_scene()
    declared = (
        ("right_first", 0.25),
        ("left_second", -0.25),
        ("right_third", 0.35),
        ("left_fourth", -0.35),
    )
    for name, x in declared:
        scene.add_monitor(mw.PointMonitor(name, (x, 0.0, 0.0), fields=("Ez",)))

    solver = _direct_distributed(scene, cuda_p2p_devices, gather_fields=False)
    output = solver.solve(
        time_steps=8,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=False,
        normalize_source=False,
        use_cuda_graph=False,
    )

    assert tuple(output["observers"]) == tuple(name for name, _ in declared)


def test_direct_distributed_multifrequency_output_preserves_frequency_metadata(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    frequencies = (0.8e9, 1.2e9)
    solver = _direct_distributed(
        _regular_scene(),
        cuda_p2p_devices,
        gather_fields=True,
    )
    output = solver.solve(
        time_steps=12,
        dft_frequency=frequencies,
        dft_window="none",
        full_field_dft=True,
        normalize_source=False,
        use_cuda_graph=False,
    )

    assert output["frequencies"] == frequencies
    for name in ("Ex", "Ey", "Ez"):
        assert output[name].shape[0] == len(frequencies)


def _ex_interface_scene(image_case: str) -> mw.Scene:
    x = np.linspace(-0.5, 0.5, 11, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 5, dtype=np.float64)
    if image_case == "y_symmetry":
        y = np.linspace(0.0, 0.4, 5, dtype=np.float64)
        boundary = mw.BoundarySpec.none()
        symmetry = (None, ("PMC", "low"), None)
        source_y = 0.0
    elif image_case == "y_periodic":
        y = np.linspace(-0.2, 0.2, 5, dtype=np.float64)
        boundary = mw.BoundarySpec.faces(default="none", y="periodic")
        symmetry = None
        source_y = -0.19
    else:
        raise ValueError(f"Unknown image case {image_case!r}.")

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
        symmetry=symmetry,
        device="cuda:0",
    )
    scene.add_source(
        mw.PointDipole(
            # The first slab owns x in [-0.5, 0); its final Ex sample is -0.05.
            position=(-0.05, source_y, 0.0),
            polarization="Ex",
            profile="ideal",
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=1.0),
            name="interface_adjacent_ex",
        )
    )
    return scene


def _distributed_component(solver, output, name: str) -> torch.Tensor:
    if name in output:
        return output[name]
    local = tuple(getattr(shard.solver, name) for shard in solver.shards)
    return solver._gather_component(name, local)


@pytest.mark.parametrize("image_case", ("y_symmetry", "y_periodic"))
def test_interface_adjacent_ideal_ex_source_image_parity(
    image_case,
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    scene = _ex_interface_scene(image_case)
    single = FDTD(prepare_scene(scene), frequency=_FREQUENCY, absorber_type="cpml")
    single.init_field()
    single.solve(
        time_steps=24,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=False,
        normalize_source=False,
        use_cuda_graph=False,
    )

    distributed = _direct_distributed(scene, cuda_p2p_devices, gather_fields=True)
    output = distributed.solve(
        time_steps=24,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=False,
        normalize_source=False,
        use_cuda_graph=False,
    )

    assert float(torch.abs(single.Ex).max().item()) > 0.0
    for name in _FIELD_NAMES:
        torch.testing.assert_close(
            _distributed_component(distributed, output, name).to("cuda:0"),
            getattr(single, name).to("cuda:0"),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
