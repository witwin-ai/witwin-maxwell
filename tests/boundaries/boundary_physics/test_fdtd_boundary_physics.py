import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw


def _run_fdtd_boundary_case(
    boundary,
    *,
    source_center=(0.0, 0.0, 0.0),
    source_width=0.05,
    domain_bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6)),
    symmetry=None,
):
    scene = mw.Scene(
        domain=mw.Domain(bounds=domain_bounds),
        grid=mw.GridSpec.uniform(0.15),
        boundary=boundary,
        symmetry=symmetry,
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=source_center,
            polarization="Ez",
            width=source_width,
            source_time=mw.CW(frequency=1e9, amplitude=25.0),
            name="src",
        )
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
        absorber="cpml",
    ).run()
    fields = {name.upper(): tensor.detach().cpu().numpy() for name, tensor in result.fields.items()}
    return scene, fields


def _run_fdtd_result(scene):
    return mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
        absorber="cpml",
    ).run()


def _build_symmetry_pair(symmetry_mode, *, polarization):
    # dl and the domain spans are exact binary fractions so span / dl carries no
    # floating-point excess and both grids mesh commensurately: the full grid
    # puts nodes exactly on the half-domain grid, including one at x = 0.
    half_scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.625), (-0.625, 0.625), (-0.625, 0.625))),
        grid=mw.GridSpec.uniform(0.125),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        symmetry=(symmetry_mode, None, None),
        device="cuda",
    )
    half_scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization=polarization,
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=25.0),
            name="src",
        )
    )

    full_scene = mw.Scene(
        # Symmetric full domain: the dipole sees identical PML distances on both
        # x faces, so the full run is the exact mirror of the folded half run.
        domain=mw.Domain(bounds=((-0.625, 0.625), (-0.625, 0.625), (-0.625, 0.625))),
        grid=mw.GridSpec.uniform(0.125),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    full_scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization=polarization,
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=25.0),
            name="src",
        )
    )
    return half_scene, full_scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_periodic_boundary_matches_opposite_faces():
    _, fields = _run_fdtd_boundary_case(mw.BoundarySpec.periodic())
    ez = fields["EZ"]
    rel_err = np.linalg.norm(ez[-1] - ez[0]) / max(np.linalg.norm(ez[0]), 1e-12)
    assert rel_err < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_bloch_boundary_matches_expected_phase_shift():
    scene, fields = _run_fdtd_boundary_case(mw.BoundarySpec.bloch((math.pi / 1.2, 0.0, 0.0)))
    ez = fields["EZ"]
    phase = scene.bloch_phase_factors[0]
    rel_err = np.linalg.norm(ez[-1] - phase * ez[0]) / max(np.linalg.norm(ez[0]), 1e-12)
    assert rel_err < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_pec_boundary_keeps_tangential_electric_field_zero():
    _, fields = _run_fdtd_boundary_case(mw.BoundarySpec.pec())
    boundary_max = max(
        np.max(np.abs(fields["EX"][:, 0, :])),
        np.max(np.abs(fields["EX"][:, -1, :])),
        np.max(np.abs(fields["EY"][0, :, :])),
        np.max(np.abs(fields["EY"][-1, :, :])),
        np.max(np.abs(fields["EZ"][0, :, :])),
        np.max(np.abs(fields["EZ"][-1, :, :])),
    )
    assert boundary_max < 1e-7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_periodic_boundary_preserves_opposite_faces_with_edge_source():
    _, fields = _run_fdtd_boundary_case(
        mw.BoundarySpec.periodic(),
        source_center=(-0.55, 0.0, 0.0),
        source_width=0.12,
    )
    ez = fields["EZ"]
    rel_err = np.linalg.norm(ez[-1] - ez[0]) / max(np.linalg.norm(ez[0]), 1e-12)
    assert rel_err < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_bloch_boundary_preserves_phase_with_edge_source():
    scene, fields = _run_fdtd_boundary_case(
        mw.BoundarySpec.bloch((math.pi / 1.2, 0.0, 0.0)),
        source_center=(-0.55, 0.0, 0.0),
        source_width=0.12,
    )
    ez = fields["EZ"]
    phase = scene.bloch_phase_factors[0]
    rel_err = np.linalg.norm(ez[-1] - phase * ez[0]) / max(np.linalg.norm(ez[0]), 1e-12)
    assert rel_err < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_mixed_periodic_and_pml_boundary_preserves_periodic_axis():
    _, fields = _run_fdtd_boundary_case(
        mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            y="periodic",
        ),
        source_center=(0.0, -0.55, 0.0),
        source_width=0.12,
        domain_bounds=((-0.9, 0.9), (-0.6, 0.6), (-0.9, 0.9)),
    )
    ez = fields["EZ"]
    rel_err = np.linalg.norm(ez[:, -1, :] - ez[:, 0, :]) / max(np.linalg.norm(ez[:, 0, :]), 1e-12)
    assert rel_err < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_mixed_bloch_xy_pml_z_runs_with_complex_fields():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            x="bloch",
            y="bloch",
            z="pml",
            bloch_wavevector=(math.pi / 1.2, math.pi / 2.4, 0.0),
        ),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.08,
            source_time=mw.CW(frequency=1.0e9, amplitude=25.0),
        )
    )
    scene.add_monitor(mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ez",)))

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=8),
        absorber="cpml",
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()

    solver = result.solver
    assert solver.complex_fields_enabled is True
    assert solver.uses_cpml is True
    value = result.monitor("center")["Ez"]
    assert torch.isfinite(torch.as_tensor(value).abs()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_mixed_bloch_pml_preserves_xy_phase_relationship():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            x="bloch",
            y="bloch",
            z="pml",
            bloch_wavevector=(math.pi / 1.2, math.pi / 2.4, 0.0),
        ),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.48, 0.0, 0.0),
            polarization="Ez",
            width=0.08,
            source_time=mw.CW(frequency=1.0e9, amplitude=40.0),
        )
    )

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=8),
        absorber="cpml",
        full_field_dft=True,
    ).run()

    ez = result.tensor("Ez")
    phase_x = complex(result.solver.boundary_phase_cos[0], result.solver.boundary_phase_sin[0])
    phase_y = complex(result.solver.boundary_phase_cos[1], result.solver.boundary_phase_sin[1])
    x_low = ez[0, ez.shape[1] // 2, ez.shape[2] // 2]
    x_high = ez[-1, ez.shape[1] // 2, ez.shape[2] // 2]
    y_low = ez[ez.shape[0] // 2, 0, ez.shape[2] // 2]
    y_high = ez[ez.shape[0] // 2, -1, ez.shape[2] // 2]
    assert torch.abs(x_high - phase_x * x_low) / torch.clamp(torch.abs(x_high), min=1e-12) < 5e-2
    assert torch.abs(y_high - phase_y * y_low) / torch.clamp(torch.abs(y_high), min=1e-12) < 5e-2

    z_cpml_memory = [
        getattr(result.solver, name)
        for name in (
            "psi_ex_z",
            "psi_ey_z",
            "psi_hx_z",
            "psi_hy_z",
            "psi_ex_z_imag",
            "psi_ey_z_imag",
            "psi_hx_z_imag",
            "psi_hy_z_imag",
        )
    ]
    assert all(bool(torch.isfinite(tensor).all().item()) for tensor in z_cpml_memory)
    assert any(torch.max(torch.abs(tensor)).item() > 0.0 for tensor in z_cpml_memory)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_mixed_bloch_cpml_rejects_two_pml_axes():
    # One Bloch axis + two PML axes is outside the single-PML-axis / two-Bloch-axes
    # contract: the split-field Bloch update carries one absorbing axis only.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.2),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=3,
            strength=1.0,
            x="bloch",
            y="pml",
            z="pml",
            bloch_wavevector=(math.pi / 1.2, 0.0, 0.0),
        ),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.1,
            source_time=mw.CW(frequency=1.0e9, amplitude=10.0),
        )
    )
    with pytest.raises(NotImplementedError, match="exactly one PML axis"):
        mw.Simulation.fdtd(
            scene,
            frequencies=[1.0e9],
            run_time=mw.TimeConfig(time_steps=2),
            absorber="cpml",
        ).run()


def _mixed_bloch_pml_scene(*, pml_axis, bloch_wavevector, source_polarization="Ez"):
    long_axis = {"x": 0, "y": 1, "z": 2}[pml_axis]
    bounds = [(-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6)]
    bounds[long_axis] = (-0.8, 0.8)
    faces = {axis: "bloch" for axis in ("x", "y", "z")}
    faces[pml_axis] = "pml"
    scene = mw.Scene(
        domain=mw.Domain(bounds=tuple(bounds)),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            bloch_wavevector=bloch_wavevector,
            **faces,
        ),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization=source_polarization,
            width=0.08,
            source_time=mw.CW(frequency=1.0e9, amplitude=40.0),
        )
    )
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize(
    ("pml_axis", "bloch_axes", "wavevector"),
    [
        ("y", ("x", "z"), (math.pi / 1.2, 0.0, math.pi / 2.4)),
        ("x", ("y", "z"), (0.0, math.pi / 1.2, math.pi / 2.4)),
    ],
)
def test_fdtd_mixed_bloch_cpml_single_pml_axis_runs(pml_axis, bloch_axes, wavevector):
    # Generalized single-PML-axis / two-Bloch-axes mixed update (beyond the
    # historical x/y-Bloch + z-PML grating layout).
    scene = _mixed_bloch_pml_scene(pml_axis=pml_axis, bloch_wavevector=wavevector)
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=8),
        absorber="cpml",
        # Force the compressed slab psi layout so the region-narrowed correction
        # kernels are exercised along the non-default absorbing axis.
        cpml_config={"memory_mode": "slab"},
        full_field_dft=True,
    ).run()
    assert result.solver._cpml_memory_mode == "slab"

    solver = result.solver
    assert solver.complex_fields_enabled is True
    assert solver.uses_cpml is True
    assert solver.has_bloch_axes == bloch_axes

    ez = result.tensor("Ez")
    assert bool(torch.isfinite(ez).all().item())

    # The absorbing axis owns the two electric and two magnetic split-field CPML
    # memory banks; they must fill (non-zero) and stay finite on both field halves.
    e_comp = {"x": ("ey", "ez"), "y": ("ex", "ez"), "z": ("ex", "ey")}[pml_axis]
    h_comp = {"x": ("hy", "hz"), "y": ("hx", "hz"), "z": ("hx", "hy")}[pml_axis]
    memory = []
    for comp in (*e_comp, *h_comp):
        for suffix in ("", "_imag"):
            memory.append(getattr(solver, f"psi_{comp}_{pml_axis}{suffix}"))
    assert all(bool(torch.isfinite(tensor).all().item()) for tensor in memory)
    assert any(torch.max(torch.abs(tensor)).item() > 0.0 for tensor in memory)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_mixed_bloch_cpml_axis_permutation_parity():
    # A y-PML scene (x/z Bloch) and an x-PML scene (y/z Bloch) related by the x<->y
    # axis swap are reflections of the same physics, so |Ez| must agree after the
    # transpose. Both run the general full-Bloch base + single-axis CPML stretch, so
    # this pins the new y- and x-PML correction kernels against each other exactly.
    wavevector = (math.pi / 1.5, 0.0, math.pi / 3.0)
    scene_y = _mixed_bloch_pml_scene(pml_axis="y", bloch_wavevector=wavevector)
    scene_x = _mixed_bloch_pml_scene(
        pml_axis="x", bloch_wavevector=(0.0, math.pi / 1.5, math.pi / 3.0)
    )
    run = dict(
        frequencies=[1.0e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=10),
        absorber="cpml",
        full_field_dft=True,
    )
    ez_y = mw.Simulation.fdtd(scene_y, **run).run().tensor("Ez")
    ez_x = mw.Simulation.fdtd(scene_x, **run).run().tensor("Ez")

    swapped = ez_y.transpose(0, 1)
    scale = torch.max(swapped.abs())
    assert float(scale) > 0.0
    rel = torch.max((ez_x.abs() - swapped.abs()).abs()) / scale
    assert float(rel) < 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_pec_boundary_keeps_tangential_electric_field_zero_with_edge_source():
    _, fields = _run_fdtd_boundary_case(
        mw.BoundarySpec.pec(),
        source_center=(-0.55, 0.0, 0.0),
        source_width=0.12,
    )
    boundary_max = max(
        np.max(np.abs(fields["EX"][:, 0, :])),
        np.max(np.abs(fields["EX"][:, -1, :])),
        np.max(np.abs(fields["EY"][0, :, :])),
        np.max(np.abs(fields["EY"][-1, :, :])),
        np.max(np.abs(fields["EZ"][0, :, :])),
        np.max(np.abs(fields["EZ"][-1, :, :])),
    )
    assert boundary_max < 1e-7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_low_face_symmetry_pec_keeps_tangential_field_zero():
    _, fields = _run_fdtd_boundary_case(
        mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        source_center=(0.15, 0.0, 0.0),
        domain_bounds=((0.0, 0.6), (-0.6, 0.6), (-0.6, 0.6)),
        symmetry=("PEC", None, None),
    )
    boundary_max = max(
        np.max(np.abs(fields["EY"][0, :, :])),
        np.max(np.abs(fields["EZ"][0, :, :])),
    )
    assert boundary_max < 1e-7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_pmc_symmetry_matches_full_domain_reference():
    half_scene, full_scene = _build_symmetry_pair("PMC", polarization="Ez")
    half_result = _run_fdtd_result(half_scene)
    full_result = _run_fdtd_result(full_scene)

    half_ez = half_result.tensor("Ez").detach().cpu().numpy()
    full_ez = full_result.tensor("Ez").detach().cpu().numpy()
    full_positive_half = full_ez[-half_ez.shape[0]:]
    rel_err = np.linalg.norm(half_ez - full_positive_half) / max(np.linalg.norm(full_positive_half), 1e-12)
    assert rel_err < 0.06


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_pec_symmetry_matches_full_domain_reference_for_normal_component():
    half_scene, full_scene = _build_symmetry_pair("PEC", polarization="Ex")
    half_result = _run_fdtd_result(half_scene)
    full_result = _run_fdtd_result(full_scene)

    half_ex = half_result.tensor("Ex").detach().cpu().numpy()
    full_ex = full_result.tensor("Ex").detach().cpu().numpy()
    full_positive_half = full_ex[-half_ex.shape[0]:]
    rel_err = np.linalg.norm(half_ex - full_positive_half) / max(np.linalg.norm(full_positive_half), 1e-12)
    # The peak pointwise difference is below 0.7%; the slightly larger global
    # norm is dominated by phase noise where the reference field is near zero.
    assert rel_err < 0.04
