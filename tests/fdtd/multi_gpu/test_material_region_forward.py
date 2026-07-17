"""Forward parity for a Box ``MaterialRegion`` under x-slab decomposition.

The distributed solver keeps the logical scene's ``MaterialRegion`` on every
shard and lets each local scene rasterize the density texture by physical
position (``grid_sample`` normalized against the region's global center/size), so
a shard's local grid selects its own sub-window with no distributed density
slicing. This test pins two things for a Box region that straddles the x-split:

* the per-shard compiled Yee permittivity, gathered back to the global grid,
  reproduces the single-GPU compile exactly (the rasterization windows stitch), and
* the gathered DFT fields and the point monitor match the single-GPU reference
  within the plan §7.2 gates.

The density is a non-trivial coarse texture (deliberately smaller than the region
grid) so ``grid_sample`` interpolation is exercised and a wrong per-shard window
would shift the interpolated permittivity at the interface.
"""

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
_TIME_STEPS = 24
_FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _density() -> torch.Tensor:
    # Coarse (4x4x4) non-uniform texture: smaller than the region grid so the
    # forward grid_sample interpolates, and asymmetric in x so a mis-placed shard
    # window would land on the wrong texel column.
    torch.manual_seed(0)
    ramp = torch.linspace(0.05, 0.95, 4)
    base = ramp[:, None, None] + 0.1 * torch.rand((4, 4, 4))
    return base.clamp(0.0, 1.0).to(dtype=torch.float32)


def _region_scene(device: str) -> mw.Scene:
    x = np.linspace(-0.5, 0.5, 11, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
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
        device=device,
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            # Straddles x = 0.0, the 6/5 partition interface of the 11-node grid.
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.3, 0.3)),
            density=_density().to(device),
            eps_bounds=(1.0, 4.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.3, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=1.0),
            name="source",
        )
    )
    scene.add_monitor(
        mw.PointMonitor("probe", (0.1, 0.0, 0.0), fields=_FIELD_NAMES)
    )
    return scene


def _parallel(devices):
    return FDTDParallelConfig(
        devices=devices,
        transport="cuda_p2p",
        overlap=True,
        gather_fields=True,
        result_device=devices[0],
    )


def _assert_field_close(actual: torch.Tensor, reference: torch.Tensor) -> None:
    actual = actual.to(reference.device)
    assert torch.isfinite(actual).all()
    absolute = torch.abs(actual - reference)
    assert float(absolute.max().item()) <= 2.0e-6
    reference_scale = float(torch.abs(reference).max().item())
    significant = torch.abs(reference) >= max(1.0e-6, 1.0e-4 * reference_scale)
    if bool(significant.any()):
        relative = absolute[significant] / torch.abs(reference[significant])
        assert float(relative.max().item()) <= 2.0e-5


def _monitor_component(payload: dict, name: str) -> torch.Tensor:
    value = payload["components"][name]
    if isinstance(value, dict):
        value = value["data"]
    return torch.as_tensor(value)


def test_distributed_box_material_region_forward_matches_single_gpu(
    cuda_p2p_devices, cuda_memory_cleanup
):
    single = FDTD(
        prepare_scene(_region_scene("cuda:0")),
        frequency=_FREQUENCY,
        absorber_type="none",
    )
    single.init_field()
    single_output = single.solve(
        time_steps=_TIME_STEPS,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=True,
        normalize_source=False,
    )

    distributed = DistributedFDTD(
        _region_scene("cuda:0"),
        frequency=_FREQUENCY,
        parallel=_parallel(cuda_p2p_devices),
        absorber_type="none",
    )
    distributed.init_field()

    # (1) The compiled permittivity must be non-trivial (the region actually bit
    # the grid) and its gathered per-shard rasterization must equal single-GPU.
    assert float((single.eps_Ez.max() - single.eps_Ez.min()).abs().item()) > 0.0
    for name in ("Ex", "Ey", "Ez"):
        eps_name = {"Ex": "eps_Ex", "Ey": "eps_Ey", "Ez": "eps_Ez"}[name]
        gathered = distributed._gather_component(
            name, tuple(getattr(shard.solver, eps_name) for shard in distributed.shards)
        )
        reference = getattr(single, eps_name)
        torch.testing.assert_close(
            gathered.to("cuda:0"), reference.to("cuda:0"), rtol=0.0, atol=0.0
        )

    distributed_output = distributed.solve(
        time_steps=_TIME_STEPS,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=True,
        normalize_source=False,
    )

    # (2) The gathered DFT fields match single-GPU within the plan field gate.
    for name in ("Ex", "Ey", "Ez"):
        _assert_field_close(distributed_output[name], single_output[name])

    # (3) The point monitor (downstream of the region) matches within the plan
    # monitor gate.
    for name in _FIELD_NAMES:
        torch.testing.assert_close(
            _monitor_component(distributed_output["observers"]["probe"], name).to("cuda:0"),
            _monitor_component(single_output["observers"]["probe"], name).to("cuda:0"),
            rtol=5.0e-5,
            atol=5.0e-6,
        )
