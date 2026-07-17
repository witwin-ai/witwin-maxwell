from __future__ import annotations

import math

import pytest
import torch
from witwin.core import GeometryBase

import witwin.maxwell as mw
from witwin.maxwell.waveport_sweep import resolve_waveport_run_manifest
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.postprocess.antenna import _far_fields_from_result


_C0 = 299792458.0
_MU0 = 1.25663706212e-6
_ETA0 = 376.730313668


class _HollowCylinder(GeometryBase):
    kind = "test_hollow_cylinder"

    def __init__(self, *, position, inner_radius, outer_radius, length, device=None):
        super().__init__(position=position, device=device)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.length = float(length)

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        radial = torch.sqrt(dy.square() + dz.square())
        axial = torch.abs(dx) - 0.5 * self.length
        outer = torch.maximum(radial - self.outer_radius, axial)
        inner = torch.maximum(radial - self.inner_radius, axial)
        return torch.maximum(outer, -inner)

    def to_mesh(self, segments=32):
        return mw.Cylinder(
            position=self.position,
            radius=self.outer_radius,
            height=self.length,
            axis="x",
            device=self.device,
        ).to_mesh(segments=segments)


def _wave_port(name="left", *, position=(-0.20, 0.0, 0.0), direction="+"):
    return mw.WavePort(
        name,
        position=position,
        size=(0.0, 0.60, 0.30),
        direction=direction,
        reference_plane=position[0],
        modes=(mw.WaveModeSpec("te", polarization="Ez"),),
    )


def _scene(*, device="cpu", ports=None, sources=()):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.4, 0.4), (-0.25, 0.25))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.none(),
        ports=(_wave_port(),) if ports is None else ports,
        sources=sources,
        device=device,
    )


def test_manifest_solves_tracks_and_normalizes_te10_channels():
    frequencies = (4.0e8, 4.5e8, 4.8e8)

    manifest = resolve_waveport_run_manifest(
        _scene(),
        mw.PortSweep(),
        frequencies,
    )

    prepared = manifest.prepared_ports[0]
    assert manifest.physical_port_names == ("left",)
    assert manifest.channel_names == ("left::TE0",)
    assert manifest.metadata()["execution"] == "single_device_frequency_sequential_cw"
    assert prepared.tracking.beta.shape == (3, 1)
    assert prepared.characteristic_impedance.shape == (1, 3)
    assert prepared.tracking_confidence.shape == (1, 3)
    assert prepared.tracking.beta.device.type == "cpu"
    assert torch.all(prepared.tracking_confidence >= 0.5)

    beta = prepared.tracking.beta[:, 0]
    expected_z = torch.as_tensor(
        [2.0 * math.pi * frequency * _MU0 for frequency in frequencies],
        dtype=torch.float64,
    ) / beta
    assert torch.allclose(
        prepared.characteristic_impedance[0].real,
        expected_z,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    cutoff = _C0 / (2.0 * 0.60)
    expected_beta = torch.as_tensor(
        [
            math.sqrt(
                (2.0 * math.pi * frequency / _C0) ** 2
                - (2.0 * math.pi * cutoff / _C0) ** 2
            )
            for frequency in frequencies
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(beta, expected_beta, rtol=0.04, atol=0.0)


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        ((), "at least one"),
        ((0.0,), "finite and positive"),
        ((5.0e8, 4.0e8), "strictly increasing"),
    ],
)
def test_manifest_rejects_invalid_frequency_axes(frequencies, message):
    with pytest.raises(ValueError, match=message):
        resolve_waveport_run_manifest(_scene(), mw.PortSweep(), frequencies)


def test_manifest_rejects_independent_sources_and_mixed_rf_port_families():
    source = mw.PointDipole(
        position=(0.0, 0.0, 0.0),
        polarization="Ez",
        source_time=mw.CW(frequency=5.0e8),
    )
    with pytest.raises(ValueError, match="independent field sources"):
        resolve_waveport_run_manifest(
            _scene(sources=(source,)),
            mw.PortSweep(),
            (5.0e8,),
        )

    lumped = mw.LumpedPort(
        "lumped",
        positive=(0.0, 0.05, 0.0),
        negative=(0.0, -0.05, 0.0),
        voltage_path=mw.AxisPath("y"),
        current_surface=mw.Box(position=(0.0, 0.0, 0.0), size=(0.10, 0.0, 0.10)),
    )
    with pytest.raises(NotImplementedError, match="explicit passive terminations"):
        resolve_waveport_run_manifest(
            _scene(ports=(_wave_port(), lumped)),
            mw.PortSweep(),
            (5.0e8,),
        )


def test_direct_waveport_excitation_validates_and_selects_a_stable_mode_name():
    excitation = mw.PortExcitation("left", mode_name="TE0")
    simulation = mw.Simulation.fdtd(
        _scene(),
        frequency=4.5e8,
        excitations=excitation,
    )

    simulation._validate_port_excitations()
    assert excitation.mode_name == "TE0"
    with pytest.raises(ValueError, match="has no mode"):
        mw.Simulation.fdtd(
            _scene(),
            frequency=4.5e8,
            excitations=mw.PortExcitation("left", mode_name="missing"),
        )._validate_port_excitations()
    with pytest.raises(ValueError, match="source_impedance='matched'"):
        mw.Simulation.fdtd(
            _scene(),
            frequency=4.5e8,
            excitations=mw.PortExcitation("left", source_impedance=50.0),
        )._validate_port_excitations()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device")
def test_manifest_keeps_mode_basis_tracking_and_impedance_on_one_cuda_device():
    manifest = resolve_waveport_run_manifest(
        _scene(device="cuda"),
        mw.PortSweep(),
        (4.2e8, 4.8e8),
    )
    prepared = manifest.prepared_ports[0]
    assert prepared.tracking.beta.device.type == "cuda"
    assert prepared.tracking.orientation.device == prepared.tracking.beta.device
    assert prepared.characteristic_impedance.device == prepared.tracking.beta.device
    assert all(
        profile.device == prepared.tracking.beta.device
        for frequency_modes in prepared.mode_data
        for mode in frequency_modes
        for profile in mode["component_profiles"].values()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device")
def test_axis_aligned_coax_tem_impedance_matches_analytic_value():
    inner_radius = 0.04
    outer_radius = 0.16
    frequency = 1.0e9
    mode = mw.WaveModeSpec(
        "tem",
        polarization="Ey",
        voltage_path=((0.0, inner_radius, 0.0), (0.0, outer_radius, 0.0)),
        current_contour=mw.Box(
            position=(0.00125, 0.0, 0.0),
            size=(0.0, 0.2025, 0.2025),
        ),
    )
    port = mw.WavePort(
        "coax",
        position=(0.0, 0.0, 0.0),
        size=(0.0, 0.40, 0.40),
        direction="+",
        reference_plane=0.0,
        modes=(mode,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.01, 0.01), (-0.205, 0.205), (-0.205, 0.205))),
        grid=mw.GridSpec.uniform(0.0025),
        boundary=mw.BoundarySpec.none(),
        ports=(port,),
        device="cuda",
    )
    scene.add_structure(
        mw.Cylinder(
            position=(0.0, 0.0, 0.0),
            radius=inner_radius,
            height=0.02,
            axis="x",
        ).with_material(mw.Material.pec(), name="inner_conductor")
    )
    scene.add_structure(
        _HollowCylinder(
            position=(0.0, 0.0, 0.0),
            inner_radius=outer_radius,
            outer_radius=0.20,
            length=0.02,
        ).with_material(mw.Material.pec(), name="outer_conductor")
    )

    prepared_scene = prepare_scene(scene)
    prepared = resolve_waveport_run_manifest(
        prepared_scene,
        mw.PortSweep(),
        (frequency,),
    ).prepared_ports[0]

    expected = _ETA0 / (2.0 * math.pi) * math.log(outer_radius / inner_radius)
    actual = float(prepared.characteristic_impedance[0, 0].real)
    assert actual == pytest.approx(expected, rel=0.02)


def _hollow_guide_scene():
    left = _wave_port("left", position=(-0.30, 0.0, 0.0), direction="+")
    right = _wave_port("right", position=(0.30, 0.0, 0.0), direction="-")
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.5, 0.5), (-0.35, 0.35))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        ports=(left, right),
        device="cuda",
    )
    wall_specs = (
        ((0.0, 0.40, 0.0), (1.0, 0.20, 0.70), "wall_y_high"),
        ((0.0, -0.40, 0.0), (1.0, 0.20, 0.70), "wall_y_low"),
        ((0.0, 0.0, 0.25), (1.0, 0.60, 0.20), "wall_z_high"),
        ((0.0, 0.0, -0.25), (1.0, 0.60, 0.20), "wall_z_low"),
    )
    for position, size, name in wall_specs:
        scene.add_structure(
            mw.Box(position=position, size=size).with_material(
                mw.Material.pec(),
                name=name,
            )
        )
    return scene


def _free_space_waveport_array_scene():
    left = _wave_port("left", position=(-0.20, 0.0, 0.0), direction="+")
    right = _wave_port("right", position=(0.20, 0.0, 0.0), direction="-")
    surface = mw.ClosedSurfaceMonitor.box(
        "array_nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.50, 0.70, 0.40),
        frequencies=(4.5e8, 4.8e8),
    )
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.4, 0.4), (-0.25, 0.25))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        ports=(left, right),
        monitors=(surface,),
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device")
def test_two_waveport_sweep_returns_physical_single_device_network():
    result = mw.Simulation.fdtd(
        _hollow_guide_scene(),
        frequencies=(4.5e8,),
        excitations=mw.PortSweep(),
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=14),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    network = result.network
    assert network is not None
    assert network.port_names == ("left::TE0", "right::TE0")
    assert network.s.shape == (1, 2, 2)
    assert network.s.device.type == "cuda"
    assert torch.all(torch.isfinite(network.s))
    assert result.stats()["network_sweep"]["execution"] == (
        "single_device_frequency_sequential_cw"
    )
    assert len(result._array_run_data.column_results) == 2
    assert all(len(column) == 1 for column in result._array_run_data.column_results)
    torch.testing.assert_close(
        result._array_run_data.incident,
        torch.stack(
            (
                result.port("left").a[0, :, 0],
                result.port("right").a[1, :, 0],
            ),
            dim=-1,
        ),
    )
    for name in ("left", "right"):
        port = result.port(name)
        assert port.mode_names == ("TE0",)
        assert port.voltage.shape == (2, 1, 1)
        assert port.beta.shape == (1, 1)
        assert port.characteristic_impedance.shape == (1, 1)
        assert port.voltage.device.type == "cuda"

    reciprocity_error = torch.abs(network.s[0, 0, 1] - network.s[0, 1, 0])
    reciprocity_scale = torch.maximum(
        torch.abs(network.s[0, 0, 1]),
        torch.abs(network.s[0, 1, 0]),
    ).clamp_min(1.0e-12)
    assert float(reciprocity_error / reciprocity_scale) < 0.10
    assert float(torch.linalg.svdvals(network.s[0]).amax()) < 1.25
    shifted_auto = network.shift_reference_planes((0.01, -0.01))
    shifted_explicit = network.shift_reference_planes(
        (0.01, -0.01),
        propagation_constants=network.metadata["propagation_constants"],
    )
    torch.testing.assert_close(shifted_auto.s, shifted_explicit.s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device")
def test_multifrequency_waveport_sweep_extracts_array_basis_without_rerun():
    scene = _free_space_waveport_array_scene()
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(4.5e8, 4.8e8),
        excitations=mw.PortSweep(),
        run_time=mw.TimeConfig(time_steps=1024),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    basis = result.array_basis(
        monitor="array_nf2ff",
        theta_points=9,
        phi_points=13,
        radius=1.0,
    )

    assert basis.port_names == ("left::TE0", "right::TE0")
    assert basis.eep.e_theta.shape == (2, 2, 9, 13)
    assert basis.dtype == torch.complex128
    assert basis.metadata["solver_rerun"] is False
    assert "complex64 to NetworkData complex128" in basis.metadata["precision_policy"]
    assert torch.equal(
        basis.radiated_power_matrix,
        basis.radiated_power_matrix.mH,
    )
    assert all(
        len(column) == 2 for column in result._array_run_data.column_results
    )
    assert all(
        not compact.ports
        for column in result._array_run_data.column_results
        for compact in column
    )

    for port_index, column in enumerate(result._array_run_data.column_results):
        for frequency_index, compact in enumerate(column):
            transformed = _far_fields_from_result(
                compact,
                surface="array_nf2ff",
                frequencies=basis.frequencies[frequency_index : frequency_index + 1],
                theta=basis.eep.theta,
                phi=basis.eep.phi,
                radius=1.0,
                phase_center=basis.eep.phase_center,
                frame=basis.eep.frame,
            )
            scale = result._array_run_data.incident[frequency_index, port_index]
            torch.testing.assert_close(
                transformed["e_theta"][0].to(dtype=basis.dtype),
                basis.eep.e_theta[frequency_index, port_index] * scale,
                rtol=2.0e-5,
                atol=1.0e-8,
            )
            torch.testing.assert_close(
                transformed["e_phi"][0].to(dtype=basis.dtype),
                basis.eep.e_phi[frequency_index, port_index] * scale,
                rtol=2.0e-5,
                atol=1.0e-8,
            )

    weights = torch.tensor(
        [[0.5 + 0.1j, -0.2 + 0.3j], [0.4 - 0.2j, 0.1 + 0.6j]],
        device=basis.device,
        dtype=basis.dtype,
    )
    beam = basis.combine(weights)
    torch.testing.assert_close(
        beam.far_field.e_theta,
        torch.einsum("fn,fntp->ftp", weights, basis.eep.e_theta),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device")
def test_prepared_direct_waveport_excitation_returns_modal_port_data_without_network():
    simulation = mw.Simulation.fdtd(
        _hollow_guide_scene(),
        frequencies=(4.5e8,),
        excitations=mw.PortExcitation("left", mode_name="left::TE0"),
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=14),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    )

    result = simulation.prepare().run()

    assert result.network is None
    assert tuple(result.ports) == ("left",)
    port = result.port("left")
    assert port.mode_names == ("TE0",)
    assert port.voltage.shape == (1, 1)
    assert port.voltage.device.type == "cuda"
    assert port.metadata["active_channel"] == "left::TE0"
    assert result.stats()["waveport_excitation"]["kind"] == (
        "direct_waveport_excitation"
    )
