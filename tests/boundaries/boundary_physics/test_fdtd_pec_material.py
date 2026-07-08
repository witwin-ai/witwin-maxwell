import numpy as np
import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD PEC material validation needs CUDA"
)


def _run_pec_slab(pec_mode, boundary):
    domain = mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)))
    grid = mw.GridSpec.uniform(0.025)
    scene = mw.Scene(
        domain=domain,
        grid=grid,
        boundary=boundary,
        device="cuda",
        subpixel_samples=mw.SubpixelSpec(pec=pec_mode),
    )
    scene.add_structure(
        mw.Structure(
            name="pec",
            geometry=mw.Box(position=(0.15, 0.0, 0.0), size=(0.2, 0.4, 0.4)),
            material=mw.Material.pec(),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.2, 0.0, 0.0),
            width=0.03,
            polarization=(0, 0, 1),
            source_time=mw.CW(frequency=5e9, amplitude=1.0),
            name="src",
        )
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(5e9,),
        run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=6),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()
    ez = np.abs(result.fields["EZ"].detach().cpu().numpy())
    x = np.linspace(-0.5, 0.5, ez.shape[0])
    iy, iz = ez.shape[1] // 2, ez.shape[2] // 2
    inside = float(ez[int(np.argmin(np.abs(x - 0.15))), iy, iz])
    source = float(ez[int(np.argmin(np.abs(x + 0.2))), iy, iz])
    return inside, source


@pytest.mark.parametrize("pec_mode", ["staircase", "conformal"])
def test_pec_slab_zeroes_tangential_interior_field(pec_mode):
    inside, source = _run_pec_slab(pec_mode, mw.BoundarySpec.pml(num_layers=8))
    assert source > 1e-2
    assert inside < 1e-3 * source


def test_interior_pec_coexists_with_boundary_pec_face():
    boundary = mw.BoundarySpec.pml(num_layers=8).with_faces(z_low="pec")
    inside, source = _run_pec_slab("conformal", boundary)
    assert source > 1e-2
    assert inside < 1e-3 * source
