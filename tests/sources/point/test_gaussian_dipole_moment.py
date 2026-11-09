import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.injection import _yee_component_control_volumes


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")


@pytest.mark.parametrize("width", (1.0e-4, 1.0e-3, 3.0e-3, 2.0e-2))
def test_open_gaussian_dipole_has_unit_moment(width):
    frequency = 1.0e9
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-4.0 * width, 4.0 * width),) * 3),
        grid=mw.GridSpec.uniform(width / 4.0),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            profile="gaussian",
            width=width,
            source_time=mw.GaussianPulse(frequency=frequency, fwidth=0.5e9),
        )
    )
    solver = mw.Simulation.fdtd(
        scene, frequencies=[frequency], run_time=mw.TimeConfig(time_steps=1)
    ).prepare().solver
    (term,) = solver._source_terms
    patch = term["patch"].double()
    region = tuple(slice(offset, offset + size) for offset, size in zip(term["offsets"], patch.shape))
    eps = solver.eps_Ez[region].double()
    control_volumes = _yee_component_control_volumes(solver, "Ez")[region].double()
    moment = float((patch * eps * control_volumes).sum()) / (-solver.dt)
    assert moment == pytest.approx(1.0, rel=1e-5, abs=0.0)
