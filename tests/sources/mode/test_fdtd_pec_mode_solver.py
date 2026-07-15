from types import SimpleNamespace
import math

import pytest
import torch

import witwin.maxwell as mw
import witwin.maxwell.fdtd.excitation.modes as mode_solver
from witwin.maxwell.compiler.sources import _compile_mode_source
from witwin.maxwell.fdtd.excitation.modes import solve_mode_source_profile
from witwin.maxwell.scene import prepare_scene


_C0 = 299792458.0


def _mode_context(scene):
    prepared = prepare_scene(scene)
    return SimpleNamespace(
        scene=prepared,
        Ex=torch.empty((1,), device=prepared.device, dtype=torch.float32),
        c=_C0,
        boundary_kind=prepared.boundary.kind,
        _compiled_material_model=prepared.compile_materials(),
        _mode_source_rebuild_from_fields=False,
    )


def _solve(scene, source):
    context = _mode_context(scene)
    compiled = _compile_mode_source(
        source,
        default_frequency=float(source.source_time.frequency),
    )
    return solve_mode_source_profile(context, compiled), context


def _square_aperture_scene(*, device, with_pec):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.none(),
        device=device,
    )
    if with_pec:
        scene.add_structure(
            mw.Cylinder(
                position=(0.0, 0.0, 0.0),
                radius=0.13,
                height=0.8,
                axis="x",
            ).with_material(mw.Material.pec(), name="inner_conductor")
        )
    return scene


def _square_aperture_source():
    return mw.ModeSource(
        position=(0.0, 0.0, 0.0),
        size=(0.0, 0.64, 0.64),
        polarization="Ey",
        source_time=mw.CW(frequency=1.0e9),
        name="mode",
    )


def test_pec_mask_changes_mode_operator_and_profile_without_scipy_material_conversion(
    monkeypatch,
):
    source = _square_aperture_source()
    baseline, _ = _solve(_square_aperture_scene(device="cpu", with_pec=False), source)

    def reject_cpu_conversion(_component):
        raise AssertionError(
            "PEC-aware mode solve must remain on the material tensor device."
        )

    monkeypatch.setattr(mode_solver, "_real_plane_numpy", reject_cpu_conversion)
    pec, context = _solve(_square_aperture_scene(device="cpu", with_pec=True), source)

    assert pec["mode_solver_kind"] == "vector_pec_dense_torch"
    assert pec["pec_node_count"] > 0
    assert pec["beta"] != pytest.approx(baseline["beta"], rel=1.0e-6, abs=1.0e-8)
    assert not torch.equal(pec["profile"], baseline["profile"])
    assert (
        pec["profile"].device
        == context._compiled_material_model["pec_occupancy"].device
    )
    assert all(
        profile.device == pec["profile"].device
        for profile in pec["component_profiles"].values()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA for device-resident eigensolve"
)
def test_pec_mode_solver_keeps_profiles_on_cuda():
    result, context = _solve(
        _square_aperture_scene(device="cuda", with_pec=True),
        _square_aperture_source(),
    )

    assert result["mode_solver_kind"] == "vector_pec_dense_torch"
    assert context._compiled_material_model["pec_occupancy"].device.type == "cuda"
    assert all(
        profile.device.type == "cuda"
        for profile in result["component_profiles"].values()
    )


def test_rectangular_pec_waveguide_te10_cutoff_beta_and_power_are_physical():
    width = 0.60
    height = 0.30
    frequency = 5.0e8
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.4, 0.4), (-0.25, 0.25))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    source = mw.ModeSource(
        position=(0.0, 0.0, 0.0),
        size=(0.0, width, height),
        polarization="Ez",
        source_time=mw.CW(frequency=frequency),
        name="te10",
    )

    mode, _ = _solve(scene, source)

    cutoff_exact = _C0 / (2.0 * width)
    k0 = 2.0 * math.pi * frequency / _C0
    beta_exact = math.sqrt(k0 * k0 - (math.pi / width) ** 2)
    cutoff_from_beta = math.sqrt(
        frequency * frequency - (mode["beta"] * _C0 / (2.0 * math.pi)) ** 2
    )
    assert abs(cutoff_from_beta - cutoff_exact) / cutoff_exact < 0.02
    assert abs(mode["beta"] - beta_exact) / beta_exact < 0.02

    electric = mode["component_profiles"]
    power_density = 0.5 * torch.real(
        electric["Ey"] * torch.conj(electric["Hz"])
        - electric["Ez"] * torch.conj(electric["Hy"])
    )
    coords_u = mode["coords_u"]
    coords_v = mode["coords_v"]
    if coords_u.numel() != power_density.shape[0]:
        offset = (coords_u.numel() - power_density.shape[0]) // 2
        coords_u = coords_u[offset : offset + power_density.shape[0]]
    if coords_v.numel() != power_density.shape[1]:
        offset = (coords_v.numel() - power_density.shape[1]) // 2
        coords_v = coords_v[offset : offset + power_density.shape[1]]
    power = torch.trapezoid(
        torch.trapezoid(power_density, x=coords_v, dim=1),
        x=coords_u,
        dim=0,
    )
    assert abs(abs(float(power)) - 1.0) < 0.01
