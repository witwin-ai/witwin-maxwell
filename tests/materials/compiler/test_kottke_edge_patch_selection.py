"""Edge-sample patches touching the last Yee plane compile exactly like the full volume."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _bulk_structures, _compile_edge_component, _static_periodic_shift_options
from witwin.maxwell.scene import prepare_scene

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _scene(grid):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))), grid=grid,
                     boundary=mw.BoundarySpec.pec(), device="cpu")
    scene.add_structure(mw.Structure(mw.Box(position=(0.13, 0.11, 0.17), size=(0.31, 0.27, 0.23)),
                                     mw.Material(eps_r=4.0)))
    return prepare_scene(scene)


@pytest.mark.parametrize("grid", [
    mw.GridSpec.uniform(0.05),
    mw.GridSpec.custom((-0.3, -0.21, -0.1, 0.0, 0.08, 0.19, 0.3), (-0.3, -0.12, 0.0, 0.1, 0.22, 0.3),
                       (-0.3, -0.18, -0.04, 0.06, 0.15, 0.3)),
], ids=["uniform", "custom"])
def test_patch_touching_the_last_plane_equals_the_full_compile(grid):
    scene = _scene(grid)
    # Same periodic-shift table the batch patch path builds.
    shifts = {id(s.geometry): _static_periodic_shift_options(scene, s.geometry) for s in _bulk_structures(scene)}
    kwargs = dict(eps_background=1.0, mu_background=1.0, averaging=scene.subpixel.averaging,
                  samples=scene.subpixel.samples, region_densities=(), geometry_periodic_shifts=shifts)
    assert scene.subpixel.averaging == "kottke_edge"
    for component in _COMPONENTS:
        full, full_sigma = _compile_edge_component(scene, component, **kwargs)
        selection = tuple(slice(max(0, n - 3), n) for n in full.shape)
        patch, patch_sigma = _compile_edge_component(scene, component, selection=selection, **kwargs)
        assert torch.equal(patch, full[selection]), component
        assert torch.equal(patch_sigma, full_sigma[selection]), component
