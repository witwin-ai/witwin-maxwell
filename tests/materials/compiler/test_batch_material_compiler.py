"""Local material sampling must preserve global Yee coordinates and gradients."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.materials import (
    _compile_edge_component, _prepare_material_region_density,
    _static_periodic_shift_options,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiler validation requires a GPU")


def prepared_scene(*, averaging, samples, nonuniform=False, periodic=False):
    axis = [-0.2, -0.16, -0.09, -0.03, 0.015, 0.08, 0.14, 0.2]
    grid = mw.GridSpec.custom(axis, axis, axis) if nonuniform else mw.GridSpec.uniform(0.04)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2),) * 3), grid=grid,
        boundary=mw.BoundarySpec.periodic() if periodic else mw.BoundarySpec.none(),
        subpixel_samples=mw.SubpixelSpec(samples=samples, averaging=averaging), device="cuda:0",
    )
    scene.add_structure(mw.Structure(
        geometry=mw.Sphere(position=(0.025, -0.01, 0.0), radius=0.13, device="cuda:0"),
        material=mw.Material(eps_r=2.5, sigma_e=0.1),
    ))
    rho = torch.linspace(0.15, 0.85, 60, device="cuda:0").reshape(4, 5, 3).requires_grad_()
    scene.add_material_region(mw.MaterialRegion(
        name="design", geometry=mw.Box(position=(0.04, 0.0, 0.0), size=(0.14, 0.18, 0.12), device="cuda:0"),
        density=rho, eps_bounds=(1.0, 4.0), mu_bounds=(1.0, 1.2),
        filter_radius=0.045, projection_beta=2.0,
    ))
    return prepare_scene(scene), rho


def compile_component(scene, component, *, selection=None):
    geometries = [s.geometry for s in scene.structures] + [r.geometry for r in scene.material_regions]
    return _compile_edge_component(
        scene, component, eps_background=1.0, mu_background=1.0,
        averaging=scene.subpixel.averaging, samples=scene.subpixel.samples,
        region_densities=tuple(_prepare_material_region_density(scene, r) for r in scene.material_regions),
        geometry_periodic_shifts={id(g): _static_periodic_shift_options(scene, g) for g in geometries},
        selection=selection,
    )


@pytest.mark.parametrize("component", ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
@pytest.mark.parametrize("averaging,samples", [("arithmetic", 1), ("arithmetic", 3), ("polarized", 3)])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_patch_values_and_density_gradients_match_full_component(component, averaging, samples, nonuniform):
    """Patch compilation equals the full component, values and density gradients.

    The former polarized samples=1 case is replaced by arithmetic samples=3
    (2026-10-29): single-sample polarized averaging raises, and the multi-sample
    arithmetic fill also exercises the coverage-fill spacing cropping on graded
    grids.
    """
    scene, rho = prepared_scene(averaging=averaging, samples=samples, nonuniform=nonuniform)
    full, full_sigma = compile_component(scene, component)
    selection = (slice(1, 5), slice(2, 6), slice(0, 3))
    patch, patch_sigma = compile_component(scene, component, selection=selection)
    torch.testing.assert_close(patch, full[selection], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(patch_sigma, full_sigma[selection], rtol=0, atol=0)
    weights = torch.linspace(0.1, 1.0, patch.numel(), device=patch.device).reshape(patch.shape)
    expected = torch.autograd.grad((full[selection] * weights).sum(), rho)[0]
    actual = torch.autograd.grad((patch * weights).sum(), rho)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_corner_patch_keeps_global_one_sided_gradient_and_thin_selection():
    scene, _ = prepared_scene(averaging="polarized", samples=3, nonuniform=True)
    full, _ = compile_component(scene, "Ex")
    for selection in [(slice(0, 1), slice(0, 1), slice(0, 1)),
                      tuple(slice(n - 1, n) for n in full.shape)]:
        patch, _ = compile_component(scene, "Ex", selection=selection)
        torch.testing.assert_close(patch, full[selection], rtol=1e-6, atol=1e-6)


def test_partial_periodic_sampling_requires_explicit_halo_routing():
    scene, _ = prepared_scene(averaging="arithmetic", samples=1, periodic=True)
    full, _ = compile_component(scene, "Ex")
    unchanged, _ = compile_component(scene, "Ex", selection=(slice(None),) * 3)
    torch.testing.assert_close(unchanged, full, rtol=0, atol=0)
    with pytest.raises(NotImplementedError, match="opposite-face"):
        compile_component(scene, "Ex", selection=(slice(1, 3),) * 3)


@pytest.mark.parametrize("selection", [
    (slice(0, 1),), (slice(-1, 3),) * 3, (slice(0, 1000),) * 3,
    (slice(0, 0),) * 3, (slice(0, 5, 2),) * 3,
])
def test_invalid_patch_selection_fails_before_sampling(selection):
    scene, _ = prepared_scene(averaging="arithmetic", samples=1)
    with pytest.raises(ValueError, match="selection"):
        compile_component(scene, "Ex", selection=selection)
