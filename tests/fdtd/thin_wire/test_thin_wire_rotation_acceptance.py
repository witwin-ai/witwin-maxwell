"""Wire-bound dipole impedance against the qualified thin-wire oracle, axial and oblique.

A 50 mm, a = 0.1 mm dipole with a 0.5 mm node-bound feed at 1 GHz runs along z at a cell
centre and along an oblique direction. Both input resistances must match the qualified CPU
integral-equation oracle (tests/support/thin_wire_mom.py, finite gap) within 3 %, the oblique
reactance within 5 %, and the two far-field patterns must agree in the dipole frame.

The axial reactance is not asserted: the thin-wire model's reactance depends on where the
wire runs in the Yee cell (known limitation: R converges; X carries a placement-dependent
bias). Measured axial X vs the oracle: about -7 to -9 % at the cell centre, up to +18 % on a
grid node line.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

import witwin.maxwell as mw
from tests.support.thin_wire_mom import dipole
from witwin.maxwell.postprocess import (
    NearFieldFarFieldTransformer,
    equivalent_surface_currents_from_monitor,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="thin-wire rotation acceptance requires CUDA"
)

_FREQUENCY = 1.0e9
_GRID_CELL_COUNTS = (64, 128)
_DOMAIN_SIZE = 0.08
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


def _scaled_point(center, direction: tuple[float, float, float], distance: float):
    return tuple(
        origin + distance * component
        for origin, component in zip(center, direction)
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
    center: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    negative_arm = torch.linspace(-0.025, -_HALF_GAP, _ARM_SEGMENTS + 1)
    positive_arm = torch.linspace(_HALF_GAP, 0.025, _ARM_SEGMENTS + 1)
    negative_points = tuple(
        _scaled_point(center, direction, distance)
        for distance in negative_arm
    )
    positive_points = tuple(
        _scaled_point(center, direction, distance)
        for distance in positive_arm
    )
    negative_wire = mw.rf.ThinWire(
        name="dipole_negative",
        points=negative_points,
        radius=1.0e-4,
        conductor=mw.rf.WireConductor.pec(),
        endpoints=(mw.rf.WireEnd.open(), mw.rf.WireEnd.open()),
        snap="continuous",
    )
    positive_wire = mw.rf.ThinWire(
        name="dipole_positive",
        points=positive_points,
        radius=1.0e-4,
        conductor=mw.rf.WireConductor.pec(),
        endpoints=(mw.rf.WireEnd.open(), mw.rf.WireEnd.open()),
        snap="continuous",
    )
    port = mw.LumpedPort(
        "feed",
        wire_binding=mw.rf.WirePortBinding.nodes(
            negative=mw.rf.WireNodeRef("dipole_negative", _ARM_SEGMENTS),
            positive=mw.rf.WireNodeRef("dipole_positive", 0),
        ),
        reference_impedance=50.0,
    )
    surface = mw.monitors.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=center,
        size=(0.07, 0.07, 0.07),
        frequencies=(_FREQUENCY,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.05, 0.05),) * 3),
        grid=mw.GridSpec.uniform(_DOMAIN_SIZE / cell_count),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        thin_wires=(negative_wire, positive_wire),
        ports=(port,),
        monitors=(surface,),
    )
    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        excitations=mw.rf.PortExcitation(
            "feed",
            source_impedance="matched",
            source_time=mw.GaussianPulse(
                frequency=_FREQUENCY,
                fwidth=0.5e9,
            ),
        ),
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig(time_steps=1),
            spectral_sampler=mw.fdtd.SpectralSampler(window="none"),
            full_field_dft=False,
        ),
    )
    prepared = simulation.prepare()
    time_steps = math.ceil(_RUN_DURATION / float(prepared.solver.dt))
    simulation.config = replace(simulation.config, run_time=mw.TimeConfig(time_steps=time_steps))
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
        port_data.z_in[0].detach().conj(),
        pattern.detach(),
        time_steps * float(result.solver.dt),
    )


def _cell_centre(cell_count: int) -> tuple[float, float, float]:
    # Grid nodes sit at integer multiples of the spacing (the domain edge is one of them).
    half = 0.5 * _DOMAIN_SIZE / cell_count
    return (half, half, _CENTER[2])


@pytest.fixture(scope="module")
def dipole_sweep():
    sweep = []
    for cell_count in _GRID_CELL_COUNTS:
        axial = _run_dipole(_AXIAL_DIRECTION, cell_count, _cell_centre(cell_count))
        oblique = _run_dipole(_OBLIQUE_DIRECTION, cell_count, _CENTER)
        sweep.append((cell_count, axial, oblique))
    return tuple(sweep)


def test_wire_bound_dipole_impedance_matches_the_qualified_oracle(dipole_sweep):
    reference = dipole(
        length=0.05, radius=1.0e-4, frequency=_FREQUENCY, gap=2.0 * _HALF_GAP, segments=640
    )
    assert reference.qualified
    oracle = reference.impedance
    for cell_count, axial, oblique in dipole_sweep:
        (axial_impedance, axial_pattern, axial_duration) = axial
        (oblique_impedance, oblique_pattern, oblique_duration) = oblique
        assert axial_duration >= _RUN_DURATION
        assert oblique_duration >= _RUN_DURATION
        for impedance in (axial_impedance, oblique_impedance):
            assert abs(float(impedance.real) / oracle.real - 1.0) <= 0.03, (cell_count, impedance)
        assert abs(float(oblique_impedance.imag) / oracle.imag - 1.0) <= 0.05, (cell_count, oblique_impedance)
        pattern_error = torch.linalg.vector_norm(
            oblique_pattern - axial_pattern
        ) / torch.linalg.vector_norm(axial_pattern)
        assert float(pattern_error) < 0.02, cell_count
