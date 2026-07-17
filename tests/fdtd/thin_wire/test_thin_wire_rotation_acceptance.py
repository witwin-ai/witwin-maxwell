from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess import (
    NearFieldFarFieldTransformer,
    equivalent_surface_currents_from_monitor,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="thin-wire rotation acceptance requires CUDA"
)

_FREQUENCY = 1.0e9
_GRID_CELL_COUNTS = (32, 64, 128)
_RUN_DURATION = 8.0e-9
_CENTER = (-6.88348921e-4, 1.32570089e-5, 1.19026851e-3)
_AXIAL_DIRECTION = (0.0, 0.0, 1.0)
_OBLIQUE_DIRECTION = (
    0.65924162,
    0.33731433,
    0.67202643,
)
_ARM_SEGMENTS = 64
_HALF_GAP = 2.5e-4


def _scaled_point(direction: tuple[float, float, float], distance: float):
    return tuple(
        origin + distance * component
        for origin, component in zip(_CENTER, direction)
    )


def _local_frame(direction: tuple[float, float, float]) -> torch.Tensor:
    local_z = torch.tensor(direction, device="cuda", dtype=torch.float64)
    local_z = local_z / torch.linalg.vector_norm(local_z)
    if torch.allclose(local_z, torch.tensor((0.0, 0.0, 1.0), device="cuda", dtype=torch.float64)):
        return torch.eye(3, device="cuda", dtype=torch.float64)
    reference = torch.tensor((0.0, 0.0, 1.0), device="cuda", dtype=torch.float64)
    local_x = torch.linalg.cross(reference, local_z)
    local_x = local_x / torch.linalg.vector_norm(local_x)
    local_y = torch.linalg.cross(local_z, local_x)
    return torch.stack((local_x, local_y, local_z), dim=1)


def _run_dipole(
    direction: tuple[float, float, float],
    cell_count: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    negative_arm = torch.linspace(-0.025, -_HALF_GAP, _ARM_SEGMENTS + 1)
    positive_arm = torch.linspace(_HALF_GAP, 0.025, _ARM_SEGMENTS + 1)
    negative_points = tuple(
        _scaled_point(direction, distance)
        for distance in negative_arm
    )
    positive_points = tuple(
        _scaled_point(direction, distance)
        for distance in positive_arm
    )
    negative_wire = mw.ThinWire(
        name="dipole_negative",
        points=negative_points,
        radius=1.0e-4,
        conductor=mw.WireConductor.pec(),
        endpoints=(mw.WireEnd.open(), mw.WireEnd.open()),
        snap="continuous",
    )
    positive_wire = mw.ThinWire(
        name="dipole_positive",
        points=positive_points,
        radius=1.0e-4,
        conductor=mw.WireConductor.pec(),
        endpoints=(mw.WireEnd.open(), mw.WireEnd.open()),
        snap="continuous",
    )
    port = mw.LumpedPort(
        "feed",
        wire_binding=mw.WirePortBinding.nodes(
            negative=mw.WireNodeRef("dipole_negative", _ARM_SEGMENTS),
            positive=mw.WireNodeRef("dipole_positive", 0),
        ),
        reference_impedance=50.0,
    )
    surface = mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=_CENTER,
        size=(0.07, 0.07, 0.07),
        frequencies=(_FREQUENCY,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.05, 0.05),) * 3),
        grid=mw.GridSpec.uniform(0.08 / cell_count),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        thin_wires=(negative_wire, positive_wire),
        ports=(port,),
        monitors=(surface,),
        device="cuda",
    )
    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        excitations=mw.PortExcitation(
            "feed",
            source_impedance="matched",
            source_time=mw.GaussianPulse(
                frequency=_FREQUENCY,
                fwidth=0.5e9,
            ),
        ),
        run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )
    prepared = simulation.prepare()
    time_steps = math.ceil(_RUN_DURATION / float(prepared.solver.dt))
    simulation.config.run_time = mw.TimeConfig(time_steps=time_steps)
    result = prepared.run()
    port_data = result.port("feed")
    theta = torch.linspace(0.0, math.pi, 13, device="cuda", dtype=torch.float64)
    phi = torch.linspace(
        0.0, 2.0 * math.pi, 17, device="cuda", dtype=torch.float64
    )
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    local_directions = torch.stack(
        (
            torch.sin(theta_grid) * torch.cos(phi_grid),
            torch.sin(theta_grid) * torch.sin(phi_grid),
            torch.cos(theta_grid),
        ),
        dim=-1,
    )
    directions = torch.einsum(
        "ij,...j->...i", _local_frame(direction), local_directions
    )
    currents = equivalent_surface_currents_from_monitor(
        result, "nf2ff", frequency=_FREQUENCY
    )
    transformed = NearFieldFarFieldTransformer(
        currents,
        solver=result.solver,
        device="cuda",
    ).transform_directions(
        directions.reshape(-1, 3),
        radius=2.0,
        batch_size=512,
    )
    electric = transformed["E"].reshape(theta_grid.shape + (3,))
    pattern = torch.sum(torch.abs(electric).square(), dim=-1)
    pattern_max = torch.amax(pattern)
    pattern = pattern / pattern_max
    return (
        port_data.z_in[0].detach(),
        pattern.detach(),
        time_steps * float(result.solver.dt),
    )


@pytest.fixture(scope="module")
def rotated_dipole_sweep():
    sweep = []
    for cell_count in _GRID_CELL_COUNTS:
        axial_impedance, axial_pattern, axial_duration = _run_dipole(
            _AXIAL_DIRECTION, cell_count
        )
        oblique_impedance, oblique_pattern, oblique_duration = _run_dipole(
            _OBLIQUE_DIRECTION, cell_count
        )
        impedance_error = torch.abs(oblique_impedance - axial_impedance) / torch.abs(
            axial_impedance
        )
        pattern_error = torch.linalg.vector_norm(
            oblique_pattern - axial_pattern
        ) / torch.linalg.vector_norm(axial_pattern)
        sweep.append(
            (
                cell_count,
                impedance_error,
                pattern_error,
                axial_duration,
                oblique_duration,
            )
        )
    return tuple(sweep)


def test_rotated_wire_bound_dipole_impedance_and_far_field_converge(
    rotated_dipole_sweep,
):
    impedance_errors = torch.stack(
        tuple(entry[1] for entry in rotated_dipole_sweep)
    )
    pattern_errors = torch.stack(tuple(entry[2] for entry in rotated_dipole_sweep))

    for cell_count, _, _, axial_duration, oblique_duration in rotated_dipole_sweep:
        assert axial_duration >= _RUN_DURATION
        assert oblique_duration >= _RUN_DURATION
    assert torch.all(torch.isfinite(impedance_errors))
    assert torch.all(torch.isfinite(pattern_errors))
    assert float(impedance_errors[-1]) < 0.02
    assert float(pattern_errors[-1]) < 0.02
    assert float(impedance_errors[-1]) < float(impedance_errors[0])
    assert float(pattern_errors[-1]) < float(pattern_errors[0])
