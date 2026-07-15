from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch
import torch.nn.functional as F

import witwin.maxwell as mw
from witwin.maxwell.fdtd.wire import (
    _target_masses,
    _validate_compiled_topology,
    deposit_wire_current,
    sample_and_update_wire,
)
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")

_FREQUENCY = 2.0e9


def _scene(
    *,
    points=None,
    radius=2.0e-3,
    source=True,
    monitor=True,
    quantities=None,
    endpoints=None,
    structures=(),
    boundary=None,
):
    if points is None:
        points = ((0.0, 0.0, -0.08), (0.0, 0.0, 0.08))
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none() if boundary is None else boundary,
        structures=structures,
        device="cuda",
    )
    scene.add_thin_wire(
        mw.ThinWire(
            name="wire",
            points=points,
            radius=radius,
            conductor=mw.WireConductor.pec(),
            endpoints=endpoints,
            snap="strict",
        )
    )
    if source:
        scene.add_source(
            mw.PointDipole(
                name="drive",
                position=(0.0, 0.0, 0.02),
                polarization="Ez",
                width=0.04,
                source_time=mw.CW(frequency=_FREQUENCY, amplitude=10.0),
            )
        )
    if monitor:
        monitor_kwargs = {}
        if quantities is not None:
            monitor_kwargs["quantities"] = quantities
        scene.add_monitor(
            mw.WireMonitor(
                name="wire_state",
                wire="wire",
                frequencies=(_FREQUENCY,),
                **monitor_kwargs,
            )
        )
    return scene


def _prepared_solver(scene, *, frequency=_FREQUENCY):
    return mw.Simulation.fdtd(
        scene,
        frequencies=(frequency,),
        run_time=mw.TimeConfig(time_steps=8),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare().solver


def _target_values(solver, components, offsets):
    values = torch.empty(offsets.numel(), device=solver.device, dtype=solver.Ex.dtype)
    for component, field in enumerate((solver.Ex, solver.Ey, solver.Ez)):
        indices = torch.nonzero(components == component, as_tuple=False).reshape(-1)
        if indices.numel():
            values.index_copy_(
                0,
                indices,
                field.reshape(-1).index_select(0, offsets.index_select(0, indices)),
            )
    return values


def _set_target_values(solver, components, offsets, values):
    for component, field in enumerate((solver.Ex, solver.Ey, solver.Ez)):
        indices = torch.nonzero(components == component, as_tuple=False).reshape(-1)
        if indices.numel():
            field.reshape(-1).index_copy_(
                0,
                offsets.index_select(0, indices),
                values.index_select(0, indices),
            )


def test_compressed_cuda_recurrence_conserves_charge_and_staggered_energy():
    solver = _prepared_solver(_scene(source=False, monitor=False))
    runtime = solver._wire_runtime
    coeff = runtime.coefficients
    masses = _target_masses(
        solver,
        coeff["target_components"],
        coeff["target_offsets"],
    )
    # One deterministic field value per owned Yee edge. No 3D wire state is used.
    initial_electric = torch.linspace(
        -2.0, 3.0, masses.numel(), device=solver.device, dtype=solver.Ex.dtype
    )
    _set_target_values(
        solver,
        coeff["target_components"],
        coeff["target_offsets"],
        initial_electric,
    )

    energies = []
    max_continuity = torch.zeros((), device=solver.device, dtype=solver.Ex.dtype)
    for _ in range(512):
        electric_before = _target_values(
            solver, coeff["target_components"], coeff["target_offsets"]
        )
        charge_before = runtime.charge.clone()
        current_before = runtime.current.clone()
        sample_and_update_wire(solver)
        current_after = runtime.current.clone()
        incidence_current = torch.zeros_like(runtime.charge)
        incidence_current.index_add_(0, coeff["tail"], current_after)
        incidence_current.index_add_(0, coeff["head"], -current_after)
        continuity = runtime.charge - charge_before + float(solver.dt) * incidence_current
        max_continuity = torch.maximum(max_continuity, continuity.abs().max())
        energies.append(
            0.5 * torch.sum(masses * electric_before.square())
            + 0.5 * torch.sum(charge_before.square() / coeff["node_capacitance"])
            + 0.5
            * torch.sum(coeff["inductance"] * current_after * current_before)
        )
        deposit_wire_current(solver)

    energy = torch.stack(energies)
    relative_drift = (energy - energy[0]).abs().max() / energy[0].abs()
    assert float(max_continuity) < 1.0e-6 * float(runtime.charge.abs().max() + 1.0e-30)
    assert float(relative_drift) < 1.0e-4
    assert runtime.current.numel() == runtime.network.segment_count
    assert runtime.charge.numel() == runtime.network.node_count
    assert runtime.state_bytes == (
        2 * runtime.network.segment_count + runtime.network.node_count
    ) * solver.Ex.element_size()
    assert float(solver.dt) < runtime.cfl_limit


def test_joint_maxwell_wire_cfl_tightens_low_frequency_step_and_stays_bounded():
    solver = _prepared_solver(
        _scene(source=False, monitor=False),
        frequency=1.0e6,
    )
    runtime = solver._wire_runtime
    assert runtime.dt_adjusted is True
    assert float(solver.dt) == pytest.approx(0.99 * runtime.cfl_limit)
    assert runtime.cfl_limit < min(
        runtime.maxwell_cfl_limit,
        runtime.wire_cfl_limit,
    )

    torch.manual_seed(710)
    fields = (solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
    for field in fields:
        field.copy_(1.0e-3 * torch.randn_like(field))
    impedance = math.sqrt(float(solver.mu0) / float(solver.eps0))
    initial_max = max(
        max(float(field.abs().max()) for field in fields[:3]),
        impedance * max(float(field.abs().max()) for field in fields[3:]),
    )
    for _ in range(600):
        _field_update_block(solver, 0.0)
    torch.cuda.synchronize()
    final_max = max(
        max(float(field.abs().max()) for field in fields[:3]),
        impedance * max(float(field.abs().max()) for field in fields[3:]),
    )
    assert math.isfinite(final_max)
    assert final_max < 10.0 * initial_max


def test_preparation_validator_rejects_unsafe_native_topology_contents():
    network = _prepared_solver(_scene(source=False, monitor=False))._wire_runtime.network

    invalid_components = network.edge_components.clone()
    invalid_components[0] = 7
    malformed_offsets = network.segment_offsets.clone()
    malformed_offsets[-1] += 1
    invalid_segments = network.contribution_segments.clone()
    invalid_segments[0] = network.segment_count
    duplicate_tail_signs = network.node_signs.clone()
    head_entry = torch.nonzero(
        (network.node_segments == 0) & (network.node_signs == -1), as_tuple=False
    ).reshape(-1)[0]
    duplicate_tail_signs[head_entry] = 1
    duplicate_offsets = network.target_offsets.clone()
    duplicate_components = network.target_components.clone()
    duplicate_offsets[1] = duplicate_offsets[0]
    duplicate_components[1] = duplicate_components[0]

    invalid_networks = (
        replace(network, edge_components=invalid_components),
        replace(network, segment_offsets=malformed_offsets),
        replace(network, contribution_segments=invalid_segments),
        replace(network, node_signs=duplicate_tail_signs),
        replace(
            network,
            target_offsets=duplicate_offsets,
            target_components=duplicate_components,
        ),
    )
    for invalid in invalid_networks:
        with pytest.raises(ValueError, match="Invalid compiled thin-wire topology"):
            _validate_compiled_topology(invalid)


def _run_forward(scene, *, cuda_graph):
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=180),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        cuda_graph=cuda_graph,
    ).run()
    return result.monitor("wire_state"), result.stats()


def test_straight_wire_forward_monitor_is_graph_exact_and_radius_sensitive():
    eager, eager_stats = _run_forward(_scene(), cuda_graph=False)
    captured, captured_stats = _run_forward(_scene(), cuda_graph=True)

    assert eager_stats["cuda_graph_active"] is False
    assert captured_stats["cuda_graph_active"] is True
    assert captured_stats["thin_wire_enabled"] is True
    assert captured_stats["thin_wire_segments"] == captured.current.shape[1]
    assert captured_stats["thin_wire_state_bytes"] > 0
    assert torch.count_nonzero(eager.current).item() > 0
    assert torch.count_nonzero(eager.charge).item() > 0
    torch.testing.assert_close(captured.current, eager.current, rtol=0.0, atol=0.0)
    torch.testing.assert_close(captured.charge, eager.charge, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(captured.ohmic_loss).item() == 0

    wider, _ = _run_forward(_scene(radius=3.0e-3), cuda_graph=False)
    assert not torch.allclose(wider.current, eager.current, rtol=1.0e-3, atol=0.0)


def test_wire_monitor_quantities_control_accumulation_and_typed_output():
    current_only, _ = _run_forward(
        _scene(quantities=("current",)), cuda_graph=False
    )

    assert current_only.current is not None
    assert current_only.charge is None
    assert current_only.ohmic_loss is None
    assert current_only.metadata["quantities"] == ("current",)


def test_l_wire_forward_propagates_current_across_the_corner():
    points = ((-0.08, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.08, 0.0))
    scene = _scene(points=points, source=False)
    scene.add_source(
        mw.PointDipole(
            name="drive",
            position=(-0.02, 0.0, 0.0),
            polarization="Ex",
            width=0.04,
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=10.0),
        )
    )
    data, stats = _run_forward(scene, cuda_graph=False)

    assert stats["thin_wire_segments"] == 4
    magnitude = data.current.abs()[0]
    assert torch.count_nonzero(magnitude[:2]).item() == 2
    assert torch.count_nonzero(magnitude[2:]).item() == 2
    assert torch.all(torch.isfinite(data.charge))


def test_named_pec_grounded_endpoint_remains_zero_during_forward():
    ground = mw.Structure(
        name="ground",
        geometry=mw.Box(position=(0.0, 0.0, -0.08), size=(0.02, 0.02, 0.02)),
        material=mw.Material.pec(),
    )
    scene = _scene(
        structures=(ground,),
        endpoints=(mw.WireEnd.grounded(structure="ground"), mw.WireEnd.open()),
    )
    data, _stats = _run_forward(scene, cuda_graph=False)

    assert torch.all(torch.isfinite(data.current))
    torch.testing.assert_close(data.charge[:, 0], torch.zeros_like(data.charge[:, 0]))


@pytest.mark.parametrize(
    ("material", "message"),
    (
        (mw.Material(eps_r=2.0, sigma_e=0.1), "conductive background"),
        (
            mw.Material(
                eps_r=2.0,
                debye_poles=(mw.DebyePole(delta_eps=1.0, tau=1.0e-10),),
            ),
            "dispersive background",
        ),
        (
            mw.Material(
                eps_r=2.0,
                nonlinearity=mw.NonlinearSusceptibility(chi3=1.0e-10),
            ),
            "nonlinear or time-modulated",
        ),
        (
            mw.Material(
                eps_r=2.0,
                modulation=mw.ModulationSpec(frequency=1.0e8, amplitude=0.1),
            ),
            "nonlinear or time-modulated",
        ),
    ),
    ids=("conductive", "dispersive", "nonlinear", "modulated"),
)
def test_phase1_rejects_unsupported_host_compositions(material, message):
    host = mw.Structure(
        name="host",
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.24, 0.24, 0.24)),
        material=material,
    )
    with pytest.raises(NotImplementedError, match=message):
        _prepared_solver(
            _scene(structures=(host,), source=False, monitor=False)
        )


def test_phase1_rejects_surface_impedance_conductor_ownership():
    surface = mw.Structure(
        name="surface_metal",
        geometry=mw.Box(position=(0.08, 0.0, 0.0), size=(0.08, 0.24, 0.24)),
        material=mw.LossyMetalMedium(conductivity=5.8e7),
    )
    with pytest.raises(NotImplementedError, match="share conductor ownership"):
        _prepared_solver(
            _scene(structures=(surface,), source=False, monitor=False)
        )


@pytest.mark.parametrize(
    "boundary",
    (mw.BoundarySpec.periodic(), mw.BoundarySpec.bloch((0.1, 0.0, 0.0))),
    ids=("periodic", "bloch"),
)
def test_phase1_rejects_periodic_wire_composition(boundary):
    with pytest.raises(NotImplementedError, match="periodic or Bloch-periodic"):
        _prepared_solver(
            _scene(boundary=boundary, source=False, monitor=False)
        )


def _half_wave_dipole_profile(segment_count: int) -> torch.Tensor:
    wavelength = 299792458.0 / _FREQUENCY
    half_length = 0.5 * wavelength
    spacing = half_length / segment_count
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half_length, half_length),) * 3),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_thin_wire(
        mw.ThinWire(
            name="dipole",
            points=((0.0, 0.0, -0.5 * half_length), (0.0, 0.0, 0.5 * half_length)),
            radius=0.02 * (half_length / 12.0),
            conductor=mw.WireConductor.pec(),
        )
    )
    # Two equal adjacent-edge drives preserve mirror symmetry about the center node.
    for index, z_position in enumerate((-0.5 * spacing, 0.5 * spacing)):
        scene.add_source(
            mw.PointDipole(
                name=f"drive_{index}",
                position=(0.0, 0.0, z_position),
                polarization="Ez",
                width=spacing,
                source_time=mw.CW(frequency=_FREQUENCY, amplitude=5.0),
            )
        )
    scene.add_monitor(
        mw.WireMonitor(
            name="dipole_state",
            wire="dipole",
            frequencies=(_FREQUENCY,),
        )
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=60 * segment_count),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    return result.monitor("dipole_state").current.abs()[0]


def test_center_driven_half_wave_dipole_forward_converges_to_sinusoidal_profile():
    profiles = {
        segment_count: _half_wave_dipole_profile(segment_count)
        for segment_count in (8, 10, 12)
    }
    errors = []
    resampled = []
    for segment_count, magnitude in profiles.items():
        centers = (
            torch.arange(segment_count, device=magnitude.device, dtype=magnitude.dtype)
            + 0.5
        ) / segment_count - 0.5
        oracle = torch.cos(math.pi * centers)
        normalized = magnitude / magnitude.max()
        errors.append(float(torch.linalg.vector_norm(normalized - oracle) / math.sqrt(segment_count)))
        resampled.append(
            F.interpolate(
                normalized.reshape(1, 1, -1),
                size=96,
                mode="linear",
                align_corners=True,
            ).reshape(-1)
        )

        assert magnitude.shape == (segment_count,)
        assert torch.all(torch.isfinite(magnitude))
        assert torch.count_nonzero(magnitude).item() == magnitude.numel()
        torch.testing.assert_close(magnitude, magnitude.flip(0), rtol=0.15, atol=0.0)
        assert magnitude[segment_count // 2 - 1 : segment_count // 2 + 1].mean() > magnitude[
            [0, -1]
        ].mean()
    coarse_change = torch.sqrt(torch.mean((resampled[1] - resampled[0]).square()))
    fine_change = torch.sqrt(torch.mean((resampled[2] - resampled[1]).square()))
    assert float(fine_change) < float(coarse_change)
    assert max(errors) < 0.15


def test_no_wire_scene_keeps_zero_state_and_no_wire_kernel_runtime():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.08, 0.08),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
    solver = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=2),
    ).prepare().solver

    assert solver._wire_runtime is None
