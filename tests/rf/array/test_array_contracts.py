import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.array import validate_array_superposition
from witwin.maxwell.array import ARRAY_ACCEPTANCE_BUDGET
from witwin.maxwell.network_sweep import resolve_network_run_manifest


def _basis(*, device="cpu", complex_z0=False, dtype=torch.complex128):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    frequencies = torch.tensor([1.0e9, 1.5e9], device=device, dtype=real_dtype)
    scattering = torch.tensor(
        [
            [[0.10 + 0.02j, 0.03 - 0.01j], [0.03 - 0.01j, -0.08 + 0.01j]],
            [[0.12 - 0.01j, 0.02 + 0.02j], [0.02 + 0.02j, -0.06 - 0.02j]],
        ],
        device=device,
        dtype=dtype,
    )
    z0 = torch.tensor(
        [[50.0 + (4.0j if complex_z0 else 0.0j), 60.0 - (3.0j if complex_z0 else 0.0j)]] * 2,
        device=device,
        dtype=dtype,
    )
    network = mw.NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=z0,
        port_names=("p1", "p2"),
    )
    theta_vector = torch.linspace(0.0, math.pi, 9, device=device, dtype=real_dtype)
    phi_vector = torch.linspace(0.0, 2.0 * math.pi, 17, device=device, dtype=real_dtype)
    theta, phi = torch.meshgrid(theta_vector, phi_vector, indexing="ij")
    element_1 = torch.sin(theta).to(dtype) * torch.exp(0.2j * torch.cos(phi))
    element_2 = torch.sin(theta).to(dtype) * torch.exp(-0.3j * torch.sin(phi))
    e_theta = torch.stack((element_1, element_2), dim=0)[None].expand(2, -1, -1, -1).clone()
    e_theta[1] *= torch.exp(torch.tensor(0.15j, device=device, dtype=dtype))
    e_phi = 0.25j * torch.flip(e_theta, dims=(-1,))
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies,
        port_names=("p1", "p2"),
        theta=theta,
        phi=phi,
        e_theta=e_theta,
        e_phi=e_phi,
        phase_center=torch.zeros(3, device=device, dtype=real_dtype),
        frame=torch.eye(3, device=device, dtype=real_dtype),
        phase_center_source="array_aabb",
    )
    return mw.ArrayBasisData(
        network=network,
        embedded_patterns=patterns,
        fingerprint="analytic-two-port",
    )


def test_phase_zero_acceptance_budget_is_frozen():
    """Pin the frozen Phase 0 thresholds so a change must be deliberate.

    This is a golden-value test only: it proves the budget has not drifted, not
    that any gate is enforced. Enforcement is covered by the tests that consume
    the budget (``test_array_fullwave.py``) and by ``benchmark/array_phase1.py``.
    """

    budget = ARRAY_ACCEPTANCE_BUDGET

    assert budget.analytic_rtol == 1.0e-6
    assert budget.contract_fdtd_complex_l2 == 5.0e-3
    assert budget.contract_fdtd_phase_rms_deg == 0.5
    assert budget.phase1_fdtd_complex_l2 == 1.0e-4
    assert budget.phase1_fdtd_phase_rms_deg == 1.0e-2
    assert budget.fdtd_phase_support_fraction == 0.10
    assert budget.port_power_relative_error == 5.0e-3
    assert budget.radiated_power_psd_relative_floor == 1.0e-9
    assert budget.active_impedance_magnitude_error == 0.05
    assert budget.active_impedance_phase_error_deg == 3.0
    assert budget.gradient_absolute_floor == 1.0e-8
    assert budget.phase1_grid_shape == (96, 96, 96)
    assert budget.phase1_pml_cells == 8
    assert budget.phase1_steps == 4096
    assert budget.phase1_beams == 16
    assert budget.phase1_angular_shape == (181, 361)
    assert budget.phase1_basis_direct_time_ratio == 0.40
    assert budget.phase1_combine_solve_time_ratio == 0.10
    assert (budget.timing_warmups, budget.timing_samples, budget.timing_order_rounds) == (
        3,
        5,
        4,
    )


def test_acceptance_budget_carries_no_cancelled_task_level_multi_gpu_scope():
    """Task-level multi-GPU was removed from scope on 2026-07-16.

    See the "Approved scope adjustment" section of
    ``docs/plans/array-active-s-mimo-implementation.md``. These fields pinned a
    device-pool scheduler and a 1/2/4-GPU scaling gate that this plan no longer
    delivers, so their absence is the contract.
    """

    for cancelled in (
        "two_gpu_parallel_efficiency",
        "four_gpu_parallel_efficiency",
        "scaling_hardware",
        "task_s_rtol",
        "task_s_atol",
        "task_basis_count",
    ):
        assert not hasattr(ARRAY_ACCEPTANCE_BUDGET, cancelled)


def test_radiated_power_operator_rejects_non_hermitian_matrix():
    """The Q_rad Hermiticity guard must actually reject a non-Hermitian operator.

    ``postprocess/array.py`` symmetrizes Q_rad before construction, so this guard
    can only ever fire for a directly constructed or deserialized operator. That
    is exactly why it needs its own coverage: the internal path cannot exercise it.
    """

    basis = _basis()
    frequency_count, port_count = basis.network.s.shape[0], basis.network.s.shape[1]
    hermitian = torch.eye(port_count, dtype=basis.dtype).expand(
        frequency_count, port_count, port_count
    ).clone()
    mw.ArrayBasisData(
        network=basis.network,
        embedded_patterns=basis.eep,
        fingerprint="hermitian-q-rad",
        radiated_power_matrix=hermitian,
    )

    skewed = hermitian.clone()
    skewed[0, 0, 1] = skewed[0, 0, 1] + 0.5j
    with pytest.raises(ValueError, match="Hermitian"):
        mw.ArrayBasisData(
            network=basis.network,
            embedded_patterns=basis.eep,
            fingerprint="non-hermitian-q-rad",
            radiated_power_matrix=skewed,
        )


def test_two_port_network_and_two_element_fields_match_direct_formulas():
    basis = _basis(complex_z0=True)
    weights = torch.tensor([0.6 + 0.1j, -0.2 + 0.7j], dtype=torch.complex128)

    beam = basis.combine(weights)

    expected_a = weights[None].expand(2, -1)
    expected_b = torch.stack(
        [basis.network.s[index] @ expected_a[index] for index in range(2)]
    )
    expected_theta = torch.stack(
        [
            sum(
                expected_a[freq, port] * basis.eep.e_theta[freq, port]
                for port in range(2)
            )
            for freq in range(2)
        ]
    )
    expected_phi = torch.stack(
        [
            sum(
                expected_a[freq, port] * basis.eep.e_phi[freq, port]
                for port in range(2)
            )
            for freq in range(2)
        ]
    )
    torch.testing.assert_close(beam.network.a, expected_a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(beam.network.b, expected_b, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(beam.far_field.E_theta, expected_theta, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(beam.far_field.E_phi, expected_phi, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(beam.network.active_reflection, expected_b / expected_a)
    gamma = expected_b / expected_a
    expected_impedance = (
        torch.conj(basis.network.z0) + basis.network.z0 * gamma
    ) / (1.0 - gamma)
    torch.testing.assert_close(beam.network.active_impedance, expected_impedance)


def test_closed_surface_power_operator_controls_absolute_radiated_power():
    base = _basis()
    power_operator = torch.tensor(
        [
            [[0.8 + 0.0j, 0.1 + 0.2j], [0.1 - 0.2j, 0.6 + 0.0j]],
            [[0.7 + 0.0j, -0.05 + 0.1j], [-0.05 - 0.1j, 0.5 + 0.0j]],
        ],
        dtype=torch.complex128,
    )
    basis = mw.ArrayBasisData(
        network=base.network,
        embedded_patterns=base.eep,
        fingerprint=base.fingerprint,
        radiated_power_matrix=power_operator,
    )
    weights = torch.tensor([0.6 + 0.1j, -0.2 + 0.7j], dtype=torch.complex128)

    beam = basis.combine(weights)
    expected = torch.real(
        torch.einsum("m,fmn,n->f", torch.conj(weights), power_operator, weights)
    )

    torch.testing.assert_close(beam.antenna.p_rad, expected)
    torch.testing.assert_close(
        beam.antenna.system_efficiency,
        expected / torch.sum(torch.abs(weights).square()),
    )
    assert beam.metadata["radiated_power_source"] == (
        "closed_surface_complex_poynting_quadratic"
    )


def test_two_hertzian_dipoles_match_pointwise_array_formula():
    device = torch.device("cpu")
    frequency = torch.tensor([1.0e9], dtype=torch.float64, device=device)
    theta_vector = torch.linspace(0.0, math.pi, 13, dtype=torch.float64, device=device)
    phi_vector = torch.linspace(0.0, 2.0 * math.pi, 25, dtype=torch.float64, device=device)
    theta, phi = torch.meshgrid(theta_vector, phi_vector, indexing="ij")
    spacing = 0.15
    wave_number = 2.0 * math.pi * frequency[0] / 299792458.0
    direction_x = torch.sin(theta) * torch.cos(phi)
    positions = (-0.5 * spacing, 0.5 * spacing)
    columns = [
        torch.sin(theta).to(torch.complex128)
        * torch.exp(-1j * wave_number * direction_x * position)
        for position in positions
    ]
    e_theta = torch.stack(columns, dim=0)[None]
    e_phi = torch.zeros_like(e_theta)
    basis = mw.ArrayBasisData(
        network=mw.NetworkData(
            frequencies=frequency,
            s=torch.zeros((1, 2, 2), dtype=torch.complex128, device=device),
            z0=50.0,
            port_names=("left", "right"),
        ),
        embedded_patterns=mw.EmbeddedElementPatternData(
            frequencies=frequency,
            port_names=("left", "right"),
            theta=theta,
            phi=phi,
            e_theta=e_theta,
            e_phi=e_phi,
            phase_center=torch.zeros(3, dtype=torch.float64, device=device),
            frame=torch.eye(3, dtype=torch.float64, device=device),
        ),
        fingerprint="two-hertzian-dipoles",
    )
    weights = torch.tensor([0.4 + 0.2j, -0.3 + 0.7j], dtype=torch.complex128)

    beam = basis.combine(weights)
    expected = torch.zeros_like(theta, dtype=torch.complex128)
    for weight, column in zip(weights, columns):
        expected = expected + weight * column

    torch.testing.assert_close(beam.far_field.e_theta[0], expected, rtol=1e-12, atol=1e-12)


def test_weight_frequency_and_batch_shapes_are_explicit():
    basis = _basis()
    frequency_weights = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 1.0j], [0.5 + 0.5j, -0.5 + 0.5j]],
        dtype=torch.complex128,
    )
    batch_weights = torch.stack((frequency_weights, torch.conj(frequency_weights)))

    frequency_beam = basis.combine(frequency_weights)
    batch_beam = basis.combine(batch_weights)

    assert frequency_beam.network.a.shape == (2, 2)
    assert frequency_beam.far_field.e_theta.shape == (2, 9, 17)
    assert batch_beam.network.a.shape == (2, 2, 2)
    assert batch_beam.far_field.e_theta.shape == (2, 2, 9, 17)
    with pytest.raises(ValueError, match="frequency-dependent.*exact shape"):
        basis.combine(torch.ones((3, 2), dtype=torch.complex128))
    with pytest.raises(TypeError, match="complex tensor"):
        basis.combine(torch.ones(2, dtype=torch.float64))


def test_zero_incident_port_returns_mask_and_nan_active_quantities():
    basis = _basis()
    beam = basis.combine(torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128))

    assert torch.all(beam.network.active_mask[:, 0])
    assert not torch.any(beam.network.active_mask[:, 1])
    assert torch.all(torch.isnan(beam.network.active_reflection[:, 1].real))
    assert torch.all(torch.isnan(beam.network.active_impedance[:, 1].real))


def test_dark_fully_reflected_beam_preserves_defined_results_and_masks_metrics():
    basis = _basis()
    reflective_network = mw.NetworkData(
        frequencies=basis.frequencies,
        s=torch.eye(2, dtype=torch.complex128)[None].expand(2, -1, -1).clone(),
        z0=basis.network.z0,
        port_names=basis.port_names,
    )
    zero_patterns = mw.EmbeddedElementPatternData(
        frequencies=basis.frequencies,
        port_names=basis.port_names,
        theta=basis.eep.theta,
        phi=basis.eep.phi,
        e_theta=torch.zeros_like(basis.eep.e_theta),
        e_phi=torch.zeros_like(basis.eep.e_phi),
        phase_center=basis.eep.phase_center,
        frame=basis.eep.frame,
    )
    dark_basis = mw.ArrayBasisData(
        network=reflective_network,
        embedded_patterns=zero_patterns,
        fingerprint="dark-reflective",
    )

    beam = dark_basis.combine(torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128))

    torch.testing.assert_close(beam.network.accepted_power, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(beam.antenna.eirp, torch.zeros(2, dtype=torch.float64))
    assert not torch.any(beam.antenna.radiation_valid)
    assert not torch.any(beam.antenna.accepted_power_valid)
    assert torch.all(torch.isnan(beam.antenna.directivity))
    assert torch.all(torch.isnan(beam.antenna.gain))
    assert torch.all(beam.antenna.realized_gain == 0.0)


def test_masked_active_impedance_does_not_poison_weight_gradient():
    basis = _basis()
    weights = torch.tensor(
        [1.0 + 0.1j, 0.0 + 0.0j],
        dtype=torch.complex128,
        requires_grad=True,
    )

    beam = basis.combine(weights)
    torch.abs(beam.network.active_impedance[:, 0]).sum().backward()

    assert weights.grad is not None
    assert torch.all(torch.isfinite(weights.grad.real))
    assert torch.all(torch.isfinite(weights.grad.imag))


def test_complex_weight_gradient_stays_in_torch_graph():
    basis = _basis()
    weights = torch.tensor(
        [0.8 + 0.1j, -0.3 + 0.4j],
        dtype=torch.complex128,
        requires_grad=True,
    )

    beam = basis.combine(weights)
    loss = beam.antenna.realized_gain.square().mean() + beam.network.accepted_power.sum()
    loss.backward()

    assert weights.grad is not None
    assert torch.all(torch.isfinite(weights.grad.real))
    assert torch.all(torch.isfinite(weights.grad.imag))


def test_complex_weight_gradient_matches_high_precision_gradcheck():
    basis = _basis()
    weights = torch.tensor(
        [0.8 + 0.1j, -0.3 + 0.4j],
        dtype=torch.complex128,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        lambda value: basis.combine(value).antenna.realized_gain.square().mean(),
        (weights,),
        eps=1e-6,
        atol=1e-5,
        rtol=0.02,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device parity")
def test_cuda_combination_preserves_device_dtype_and_analytic_parity():
    cpu_basis = _basis(dtype=torch.complex64)
    basis = _basis(device="cuda", dtype=torch.complex64)
    weights = torch.tensor(
        [0.7 - 0.2j, 0.1 + 0.6j],
        device="cuda",
        dtype=torch.complex64,
    )

    beam = basis.combine(weights)
    expected = cpu_basis.combine(weights.cpu())

    assert beam.device.type == "cuda"
    assert beam.network.b.dtype == torch.complex64
    torch.testing.assert_close(
        beam.network.b.cpu(), expected.network.b, rtol=2e-5, atol=1e-6
    )


def _port(name, *, x=0.0, reference_impedance=50.0):
    return mw.LumpedPort(
        name=name,
        negative=(x, 0.0, -0.1),
        positive=(x, 0.0, 0.1),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(x, 0.0, 0.0), size=(0.2, 0.2, 0.0)),
        reference_impedance=reference_impedance,
    )


def test_scene_compile_array_monitors_freezes_port_order_grid_and_phase_center():
    surface = mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.8, 0.8, 0.8),
        frequencies=(1.0e9,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        ports=(_port("p1"), _port("p2")),
        monitors=(surface,),
        structures=(
            mw.Structure(
                name="array_body",
                geometry=mw.Box(position=(0.1, -0.1, 0.0), size=(0.4, 0.2, 0.1)),
                material=mw.Material(eps_r=2.0),
            ),
        ),
        device="cpu",
    )
    theta = torch.linspace(0.0, math.pi, 7)
    phi = torch.linspace(0.0, 2.0 * math.pi, 11)

    (request,) = scene.compile_array_monitors(monitor="nf2ff", theta=theta, phi=phi)

    assert request.port_names == ("p1", "p2")
    assert request.phase_center_source == "array_aabb"
    torch.testing.assert_close(request.phase_center, torch.zeros(3, dtype=torch.float32))
    assert request.theta.shape == (7, 11)
    assert len(request.monitor_faces) == 6
    with pytest.raises(ValueError, match="exactly match"):
        scene.compile_array_monitors(monitor="nf2ff", frequencies=(1.1e9,))
    with pytest.raises(TypeError, match="configuration tensors must use dtype"):
        scene.compile_array_monitors(
            monitor="nf2ff",
            theta=theta,
            phi=phi,
            dtype=torch.float64,
        )


def test_compile_array_request_consumes_selected_network_manifest_order():
    surface = mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.8, 0.8, 0.8),
        frequencies=(1.0e9,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        ports=(_port("p1"), _port("p2")),
        monitors=(surface,),
        device="cpu",
    )
    manifest = resolve_network_run_manifest(
        scene,
        mw.PortSweep(ports=("p2",)),
        (1.0e9,),
    )

    (request,) = scene.compile_array_monitors(
        monitor="nf2ff",
        run_manifest=manifest,
        theta=torch.linspace(0.0, math.pi, 5),
        phi=torch.linspace(0.0, 2.0 * math.pi, 7),
    )

    assert request.port_names == ("p2",)
    assert request.physical_port_names == ("p1", "p2")
    assert request.frequencies == manifest.frequencies
    assert request.run_manifest_metadata == manifest.metadata()
    assert len(request.run_manifest_fingerprint) == 64


def test_compile_array_request_rejects_manifest_from_different_scene_contract():
    surface = mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.8, 0.8, 0.8),
        frequencies=(1.0e9,),
    )
    current = mw.Scene(
        ports=(_port("p1"),),
        monitors=(surface,),
        device="cpu",
    )
    foreign_port = _port("p1", reference_impedance=75.0)
    foreign = mw.Scene(ports=(foreign_port,), device="cpu")
    manifest = resolve_network_run_manifest(foreign, mw.PortSweep(), (1.0e9,))

    with pytest.raises(ValueError, match="does not match this Scene"):
        current.compile_array_monitors(
            monitor="nf2ff",
            run_manifest=manifest,
            phase_center=(0.0, 0.0, 0.0),
            theta=torch.linspace(0.0, math.pi, 5),
            phi=torch.linspace(0.0, 2.0 * math.pi, 7),
        )


def test_default_phase_center_is_invariant_to_selected_basis_subset():
    surface = mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.8, 0.8, 0.8),
        frequencies=(1.0e9,),
    )
    scene = mw.Scene(
        ports=(_port("left", x=-0.2), _port("right", x=0.2)),
        monitors=(surface,),
        device="cpu",
    )
    manifest = resolve_network_run_manifest(
        scene,
        mw.PortSweep(ports=("right",)),
        (1.0e9,),
    )

    (request,) = scene.compile_array_monitors(
        monitor="nf2ff",
        run_manifest=manifest,
        theta=torch.linspace(0.0, math.pi, 5),
        phi=torch.linspace(0.0, 2.0 * math.pi, 7),
    )

    assert request.port_names == ("right",)
    assert request.physical_port_names == ("left", "right")
    torch.testing.assert_close(request.phase_center, torch.zeros(3, dtype=torch.float32))


def test_compile_array_request_rejects_missing_monitor_port_and_invalid_angles():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        ports=(_port("p1"),),
        device="cpu",
    )
    with pytest.raises(ValueError, match="no ClosedSurfaceMonitor"):
        scene.compile_array_monitors()
    surface = mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.8, 0.8, 0.8),
        frequencies=(1.0e9,),
    )
    scene.add_monitor(surface)
    with pytest.raises(ValueError, match="non-RF or missing"):
        scene.compile_array_monitors(monitor="nf2ff", ports=("missing",))
    with pytest.raises(ValueError, match="theta must span"):
        scene.compile_array_monitors(
            monitor="nf2ff",
            theta=torch.linspace(0.1, math.pi, 5),
            phi=torch.linspace(0.0, 2.0 * math.pi, 7),
        )


def test_source_bearing_mode_port_is_rejected_before_basis_compilation():
    surface = mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.8, 0.8, 0.8),
        frequencies=(1.0e9,),
    )
    mode_port = mw.ModePort(
        "driven_mode",
        position=(0.0, 0.0, 0.0),
        size=(0.2, 0.2, 0.0),
        source_time=mw.CW(frequency=1.0e9),
    )
    scene = mw.Scene(ports=(mode_port,), monitors=(surface,), device="cpu")

    with pytest.raises(ValueError, match="driven_mode::source.*independent field source"):
        scene.compile_array_monitors(monitor="nf2ff")


def test_linearity_guard_reports_named_nonlinear_and_time_varying_objects():
    nonlinear = mw.Structure(
        name="nonlinear_insert",
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
        material=mw.Material(eps_r=2.0, kerr_chi3=1.0e-18),
    )
    modulated = mw.Structure(
        name="modulated_insert",
        geometry=mw.Box(position=(0.2, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
        material=mw.Material(
            eps_r=2.0,
            modulation=mw.ModulationSpec(frequency=1.0e8, amplitude=0.05),
        ),
    )
    scene = mw.Scene(structures=(nonlinear, modulated), device="cpu")

    with pytest.raises(ValueError, match="nonlinear_insert.*modulated_insert"):
        validate_array_superposition(scene)


def test_array_basis_rejects_mismatched_port_frequency_and_wave_conventions():
    basis = _basis()
    reversed_patterns = mw.EmbeddedElementPatternData(
        frequencies=basis.frequencies,
        port_names=tuple(reversed(basis.port_names)),
        theta=basis.eep.theta,
        phi=basis.eep.phi,
        e_theta=basis.eep.e_theta[:, [1, 0]],
        e_phi=basis.eep.e_phi[:, [1, 0]],
        phase_center=basis.eep.phase_center,
        frame=basis.eep.frame,
    )
    with pytest.raises(ValueError, match="identical port order"):
        mw.ArrayBasisData(
            network=basis.network,
            embedded_patterns=reversed_patterns,
            fingerprint="bad-order",
        )
    bad_network = mw.NetworkData(
        frequencies=basis.frequencies,
        s=basis.network.s,
        z0=basis.network.z0,
        port_names=basis.port_names,
        power_wave_convention="voltage waves",
    )
    with pytest.raises(ValueError, match="power-wave convention"):
        mw.ArrayBasisData(
            network=bad_network,
            embedded_patterns=basis.eep,
            fingerprint="bad-wave-convention",
        )
    with pytest.raises(ValueError, match="power_normalization"):
        mw.EmbeddedElementPatternData(
            frequencies=basis.frequencies,
            port_names=basis.port_names,
            theta=basis.eep.theta,
            phi=basis.eep.phi,
            e_theta=basis.eep.e_theta,
            e_phi=basis.eep.e_phi,
            phase_center=basis.eep.phase_center,
            frame=basis.eep.frame,
            power_normalization="voltage normalized",
        )


def test_embedded_pattern_rejects_invalid_metadata_and_nonmonotonic_frequency():
    basis = _basis()
    kwargs = dict(
        frequencies=basis.frequencies,
        port_names=basis.port_names,
        theta=basis.eep.theta,
        phi=basis.eep.phi,
        e_theta=basis.eep.e_theta,
        e_phi=basis.eep.e_phi,
        phase_center=basis.eep.phase_center,
        frame=basis.eep.frame,
    )
    with pytest.raises(TypeError, match="wave_impedance must be real"):
        mw.EmbeddedElementPatternData(**kwargs, wave_impedance=377.0 + 1.0j)
    with pytest.raises(ValueError, match="phase_center and frame must contain"):
        mw.EmbeddedElementPatternData(
            **{**kwargs, "phase_center": torch.tensor([float("nan"), 0.0, 0.0])}
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        mw.EmbeddedElementPatternData(
            **{
                **kwargs,
                "frequencies": torch.flip(basis.frequencies, dims=(0,)),
                "e_theta": torch.flip(basis.eep.e_theta, dims=(0,)),
                "e_phi": torch.flip(basis.eep.e_phi, dims=(0,)),
            }
        )
    with pytest.raises(ValueError, match="non-empty frequency and port axes"):
        mw.EmbeddedElementPatternData(
            frequencies=torch.empty(0, dtype=torch.float64),
            port_names=(),
            theta=basis.eep.theta[:3, :2],
            phi=basis.eep.phi[:3, :2],
            e_theta=torch.empty((0, 0, 3, 2), dtype=torch.complex128),
            e_phi=torch.empty((0, 0, 3, 2), dtype=torch.complex128),
            phase_center=basis.eep.phase_center,
            frame=basis.eep.frame,
        )


def test_array_types_are_top_level_public_api():
    assert mw.ArrayBasisData.__module__ == "witwin.maxwell.array"
    assert mw.EmbeddedElementPatternData.__module__ == "witwin.maxwell.array"
    assert mw.BeamWeights.__module__ == "witwin.maxwell.array"
