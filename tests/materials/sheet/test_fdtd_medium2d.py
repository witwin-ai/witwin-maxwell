import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from tests.materials.sheet.sheet_measurement import Z0 as _Z0, measure_sheet_scattering
from witwin.core import Box
from witwin.core.material import VACUUM_PERMITTIVITY
from witwin.maxwell.scene import prepare_scene


def _build_cpu_sheet_scene(*, material, size=(0.0, 2.0, 2.0), position=(0.0, 0.0, 0.0)):
    return prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
            grid=mw.GridSpec.uniform(0.05),
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            device="cpu",
            structures=[
                mw.Structure(geometry=Box(position=position, size=size), material=material)
            ],
        )
    )


def test_medium2d_construction_and_sheet_conductivity():
    sheet = mw.Medium2D(sigma_s=2.5e-3, name="sheet")
    assert sheet.is_medium2d
    assert sheet.sigma_s == pytest.approx(2.5e-3)
    assert sheet.eps_r == pytest.approx(1.0)
    assert sheet.sigma_e == pytest.approx(0.0)
    assert sheet.sheet_pole_terms() == ()
    assert sheet.sheet_conductivity(2.0 * np.pi * 1.0e9) == pytest.approx(2.5e-3)
    assert sheet.sheet_conductivity_at_freq(1.0e9) == pytest.approx(2.5e-3)

    with pytest.raises(ValueError):
        mw.Medium2D(sigma_s=-1.0e-3)


def test_medium2d_compiles_to_single_plane_tangential_conductivity():
    sigma_s = 2.0e-3
    scene = _build_cpu_sheet_scene(material=mw.Medium2D(sigma_s=sigma_s))
    model = scene.compile_materials()

    node_index = int(np.argmin(np.abs(scene.x_nodes64 - 0.0)))
    dual = float(scene.dx_dual64[node_index])
    expected = sigma_s / dual

    for axis in ("y", "z"):
        sigma = model["sigma_e_components"][axis]
        plane = sigma[node_index]
        assert torch.allclose(plane, torch.full_like(plane, expected), rtol=1e-6)
        off_plane = torch.cat([sigma[:node_index], sigma[node_index + 1 :]])
        assert torch.all(off_plane == 0)
    assert torch.all(model["sigma_e_components"]["x"] == 0)
    # The sheet must not perturb the static permittivity or permeability.
    for axis in ("x", "y", "z"):
        assert torch.all(model["eps_components"][axis] == 1.0)
        assert torch.all(model["mu_components"][axis] == 1.0)


def test_medium2d_sheet_is_applied_once_with_subpixel_averaging():
    sigma_s = 2.0e-3
    scene = _build_cpu_sheet_scene(material=mw.Medium2D(sigma_s=sigma_s))
    model = scene.compile_materials(subpixel_samples=(2, 2, 2))

    node_index = int(np.argmin(np.abs(scene.x_nodes64 - 0.0)))
    dual = float(scene.dx_dual64[node_index])
    plane = model["sigma_e_components"]["y"][node_index]
    assert torch.allclose(plane, torch.full_like(plane, sigma_s / dual), rtol=1e-6)


def test_medium2d_frequency_evaluation_adds_conductivity_term():
    sigma_s = 2.0e-3
    frequency = 1.0e9
    scene = _build_cpu_sheet_scene(material=mw.Medium2D(sigma_s=sigma_s))
    eps_components, _ = scene.compile_material_components(frequency=frequency)

    node_index = int(np.argmin(np.abs(scene.x_nodes64 - 0.0)))
    dual = float(scene.dx_dual64[node_index])
    omega = 2.0 * np.pi * frequency
    expected_imag = -(sigma_s / dual) / (omega * VACUUM_PERMITTIVITY)

    plane = eps_components["y"][node_index]
    assert torch.allclose(plane.imag, torch.full_like(plane.imag, expected_imag), rtol=1e-5)
    assert torch.allclose(plane.real, torch.ones_like(plane.real), rtol=1e-6)


def test_medium2d_requires_box_with_exactly_one_zero_axis():
    thick = _build_cpu_sheet_scene(material=mw.Medium2D(sigma_s=1.0e-3), size=(0.2, 0.3, 0.3))
    with pytest.raises(ValueError, match="exactly one zero-size axis"):
        thick.compile_materials()

    line = _build_cpu_sheet_scene(material=mw.Medium2D(sigma_s=1.0e-3), size=(0.0, 0.0, 0.3))
    with pytest.raises(ValueError, match="exactly one zero-size axis"):
        line.compile_materials()

    sphere_scene = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
            grid=mw.GridSpec.uniform(0.05),
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            device="cpu",
            structures=[
                mw.Structure(geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.1), material=mw.Medium2D(sigma_s=1.0e-3))
            ],
        )
    )
    with pytest.raises(NotImplementedError, match="Box structure geometry"):
        sphere_scene.compile_materials()


def test_medium2d_is_rejected_by_adjoint_bridges():
    from witwin.maxwell.fdfd.adjoint.bridge import _unsupported_fdfd_adjoint_medium
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    scene = _build_cpu_sheet_scene(material=mw.Medium2D(sigma_s=1.0e-3))
    fdtd_message = _unsupported_adjoint_medium(scene)
    fdfd_message = _unsupported_fdfd_adjoint_medium(scene)
    assert fdtd_message is not None and "Medium2D" in fdtd_message
    assert fdfd_message is not None and "Medium2D" in fdfd_message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_medium2d_sheet_matches_analytic_fresnel():
    # Free-standing conductive sheet at normal incidence:
    # t = 2 / (2 + Z0*sigma_s), r = -Z0*sigma_s / (2 + Z0*sigma_s).
    frequency = 5.0e8
    for z0_sigma in (0.5, 1.0):
        sigma_s = z0_sigma / _Z0
        t, r = measure_sheet_scattering(
            mw.Medium2D(sigma_s=sigma_s),
            frequency,
            dl=0.025,
            half_length=1.0,
            half_width=0.4,
            fit_gap=0.08,
            fit_extent=0.6,
        )
        t_exact = 2.0 / (2.0 + z0_sigma)
        r_exact = z0_sigma / (2.0 + z0_sigma)
        assert abs(abs(t) - t_exact) < 0.01
        assert abs(abs(r) - r_exact) < 0.01
