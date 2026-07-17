import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.array import ARRAY_ACCEPTANCE_BUDGET
from witwin.maxwell.postprocess.antenna import _far_fields_from_result


def _port(name: str, x: float) -> mw.LumpedPort:
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.005),
        negative=(x, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(x, 0.0, -0.0025),
            size=(0.005, 0.015, 0.0),
        ),
        reference_impedance=50.0,
    )


def _scene(element_count: int) -> mw.Scene:
    positions = (torch.arange(element_count, dtype=torch.float64) - 0.5 * (element_count - 1))
    positions *= 0.01
    surface = mw.ClosedSurfaceMonitor.box(
        "array_nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.05, 0.05, 0.05),
        frequencies=(2.5e9,),
    )
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.04, 0.04),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        ports=tuple(_port(f"p{index + 1}", float(x)) for index, x in enumerate(positions)),
        monitors=(surface,),
        device="cuda",
    )


def _simulation(scene, excitations):
    return mw.Simulation.fdtd(
        scene,
        frequency=2.5e9,
        excitations=excitations,
        run_time=mw.TimeConfig(time_steps=192),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )


def _complex_l2(actual_theta, actual_phi, expected_theta, expected_phi, theta) -> float:
    weights = torch.sin(theta)
    numerator = torch.sum(
        weights * (torch.abs(actual_theta - expected_theta).square() + torch.abs(actual_phi - expected_phi).square())
    )
    denominator = torch.sum(
        weights * (torch.abs(expected_theta).square() + torch.abs(expected_phi).square())
    )
    return float(torch.sqrt(numerator / denominator))


def _phase_rms_degrees(actual_theta, actual_phi, expected_theta, expected_phi) -> float:
    expected_magnitude = torch.sqrt(
        torch.abs(expected_theta).square() + torch.abs(expected_phi).square()
    )
    support = expected_magnitude >= (
        ARRAY_ACCEPTANCE_BUDGET.fdtd_phase_support_fraction * torch.amax(expected_magnitude)
    )
    coherent_product = (
        actual_theta * torch.conj(expected_theta)
        + actual_phi * torch.conj(expected_phi)
    )
    phase = torch.angle(coherent_product[support])
    return float(torch.sqrt(torch.mean(phase.square())) * 180.0 / math.pi)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device")
@pytest.mark.parametrize("element_count", (2, 4))
@pytest.mark.parametrize("steering", ("broadside", "endfire"))
def test_fullwave_basis_matches_direct_multi_source_fdtd(element_count, steering):
    scene = _scene(element_count)
    source_time = mw.GaussianPulse(frequency=2.5e9, fwidth=1.0e9, delay=0.3e-9)
    sweep = mw.PortSweep(source_time=source_time)
    result = _simulation(scene, sweep).run()
    retained = result._array_run_data.column_results
    assert all(not hasattr(column[0].solver, "Ex") for column in retained)
    assert len({id(column[0].prepared_scene) for column in retained}) == 1
    assert all(not column[0].ports for column in retained)
    retained_monitor_names = {
        face.name for monitor in scene.monitors for face in monitor.faces
    }
    assert all(
        set(column[0].monitors) == retained_monitor_names for column in retained
    )
    basis = result.array_basis(
        monitor="array_nf2ff",
        theta_points=31,
        phi_points=61,
        radius=1.0,
    )

    # Q_rad is symmetrized on construction, so asserting Hermiticity here would be
    # tautological and prove nothing. Positive semidefiniteness is the property
    # that carries physics: real(a^H Q_rad a) is radiated power, so a negative
    # eigenvalue means some excitation radiates negative power. With 4 PML layers
    # the NF2FF box is far enough from the boundary that the closed-surface
    # Poynting operator is genuinely PSD, so the smallest eigenvalue is gated at
    # the largest only up to a floating-point roundoff band (2 PML layers put it
    # ~1 cell from the boundary and reflection drove min_eig to -2.833e-5).
    eigenvalues = torch.linalg.eigvalsh(basis.radiated_power_matrix)
    largest = torch.amax(eigenvalues)
    assert float(largest) > 0.0
    assert float(torch.amin(eigenvalues)) >= (
        -ARRAY_ACCEPTANCE_BUDGET.radiated_power_psd_relative_floor * float(largest)
    )

    drive_amplitudes = torch.ones(element_count, device=basis.device, dtype=basis.dtype)
    if steering == "endfire":
        spacing = 0.01
        wave_number = 2.0 * math.pi * 2.5e9 / 299792458.0
        positions = (
            torch.arange(element_count, device=basis.device, dtype=basis.frequencies.dtype)
            - 0.5 * (element_count - 1)
        ) * spacing
        drive_amplitudes = torch.exp(1j * wave_number * positions).to(dtype=basis.dtype)
    excitations = tuple(
        mw.PortExcitation(
            name,
            amplitude=amplitude.detach().cpu(),
            source_impedance="matched",
            source_time=source_time,
        )
        for name, amplitude in zip(basis.port_names, drive_amplitudes)
    )
    direct = _simulation(scene, excitations).run()
    transformed = _far_fields_from_result(
        direct,
        surface="array_nf2ff",
        frequencies=basis.frequencies,
        theta=basis.eep.theta,
        phi=basis.eep.phi,
        radius=1.0,
        phase_center=basis.eep.phase_center,
        frame=basis.eep.frame,
    )
    direct_a = torch.stack([direct.port(name).a for name in basis.port_names], dim=-1)
    direct_b = torch.stack([direct.port(name).b for name in basis.port_names], dim=-1)
    beam = basis.combine(direct_a)
    adjacent_phase = torch.angle(direct_a[:, 1:] / direct_a[:, :-1])
    target_phase = 0.0 if steering == "broadside" else wave_number * spacing
    torch.testing.assert_close(
        adjacent_phase,
        torch.full_like(adjacent_phase, target_phase),
        rtol=0.0,
        atol=math.radians(3.0),
    )
    direct_incident_per_port = torch.abs(direct_a).square()
    direct_reflected_per_port = torch.abs(direct_b).square()
    direct_accepted = torch.sum(
        direct_incident_per_port - direct_reflected_per_port,
        dim=-1,
    )
    torch.testing.assert_close(
        direct_incident_per_port,
        beam.network.incident_power_per_port,
        rtol=ARRAY_ACCEPTANCE_BUDGET.port_power_relative_error,
        atol=1.0e-8,
    )
    torch.testing.assert_close(
        direct_reflected_per_port,
        beam.network.reflected_power_per_port,
        rtol=ARRAY_ACCEPTANCE_BUDGET.port_power_relative_error,
        atol=1.0e-8,
    )
    torch.testing.assert_close(
        direct_accepted,
        beam.network.accepted_power,
        rtol=ARRAY_ACCEPTANCE_BUDGET.port_power_relative_error,
        atol=1.0e-8,
    )

    assert _complex_l2(
        transformed["e_theta"],
        transformed["e_phi"],
        beam.far_field.e_theta,
        beam.far_field.e_phi,
        basis.eep.theta,
    ) <= (
        ARRAY_ACCEPTANCE_BUDGET.contract_fdtd_complex_l2
    )
    assert _phase_rms_degrees(
        transformed["e_theta"],
        transformed["e_phi"],
        beam.far_field.e_theta,
        beam.far_field.e_phi,
    ) <= (
        ARRAY_ACCEPTANCE_BUDGET.contract_fdtd_phase_rms_deg
    )

    assert basis.metadata["normalization_incident_wave"] == "measured a_n(f)"
    assert basis.metadata["solver_rerun"] is False
    if element_count == 2 and steering == "broadside":
        with pytest.raises(ValueError, match="configuration tensors must be on device"):
            result.array_basis(
                monitor="array_nf2ff",
                theta=torch.linspace(0.0, math.pi, 5, dtype=basis.eep.theta.dtype),
                phi=torch.linspace(
                    0.0, 2.0 * math.pi, 7, dtype=basis.eep.theta.dtype
                ),
            )
        with pytest.raises(ValueError, match="radius tensor must be on device"):
            result.array_basis(
                monitor="array_nf2ff",
                radius=torch.tensor(1.0, dtype=basis.eep.theta.dtype),
            )
        wrong_dtype = (
            torch.float32
            if basis.frequencies.dtype == torch.float64
            else torch.float64
        )
        with pytest.raises(TypeError, match="configuration tensors must use dtype"):
            result.array_basis(
                monitor="array_nf2ff",
                theta=torch.linspace(0.0, math.pi, 5, device="cuda", dtype=wrong_dtype),
                phi=torch.linspace(
                    0.0, 2.0 * math.pi, 7, device="cuda", dtype=wrong_dtype
                ),
            )
        with pytest.raises(TypeError, match="configuration tensors must use dtype"):
            result.array_basis(
                monitor="array_nf2ff",
                phase_center=torch.zeros(3, device="cuda", dtype=wrong_dtype),
            )
        with pytest.raises(TypeError, match="configuration tensors must use dtype"):
            result.array_basis(
                monitor="array_nf2ff",
                frame=torch.eye(3, device="cuda", dtype=wrong_dtype),
            )
        with pytest.raises(TypeError, match="radius tensor must use real dtype"):
            result.array_basis(
                monitor="array_nf2ff",
                radius=torch.tensor(1.0, device="cuda", dtype=wrong_dtype),
            )

        polarized = result.array_basis(
            monitor="array_nf2ff",
            theta=basis.eep.theta,
            phi=basis.eep.phi,
            polarization=mw.Ludwig3(reference_angle=0.5),
        )
        farther = result.array_basis(
            monitor="array_nf2ff",
            theta=basis.eep.theta,
            phi=basis.eep.phi,
            radius=2.0,
        )
        assert polarized.fingerprint != basis.fingerprint
        assert farther.fingerprint != basis.fingerprint


def test_direct_multi_source_contract_rejects_waveport_mixing():
    scene = mw.Scene(
        ports=(
            mw.WavePort(
                "wave",
                position=(0.0, 0.0, 0.0),
                size=(0.0, 0.02, 0.02),
                direction="+",
                reference_plane=0.0,
                modes=(mw.WaveModeSpec("te", polarization="Ez"),),
            ),
            _port("lumped", 0.01),
        ),
        device="cpu",
    )
    simulation = mw.Simulation.fdtd(
        scene,
        frequency=1.0e9,
        excitations=(mw.PortExcitation("wave"), mw.PortExcitation("lumped")),
    )

    with pytest.raises(NotImplementedError, match="WavePort channels require"):
        simulation._validate_port_excitations()
