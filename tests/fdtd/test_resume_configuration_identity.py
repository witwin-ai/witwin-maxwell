"""Resume identity covers constitutive laws and excludes recomputed scratch."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.resume import _configuration_fingerprint
from tests.fdtd.test_time_step_stability import _simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_resume_identity_distinguishes_equal_shape_ade_coefficients():
    first = _simulation(mw.Material.debye(eps_inf=2., delta_eps=1., tau=1e-9), dt=1e-12).prepare().solver
    second = _simulation(mw.Material.debye(eps_inf=2., delta_eps=1., tau=2e-9), dt=1e-12).prepare().solver
    assert torch.equal(first.eps_Ex, second.eps_Ex)
    assert first.dt == second.dt
    assert _configuration_fingerprint(first) != _configuration_fingerprint(second)


def test_nonlinear_update_work_buffers_do_not_change_resume_identity():
    solver = _simulation(mw.Material(eps_r=2., nonlinearity=[mw.NonlinearSusceptibility(chi2=1e-4)]), dt=1e-12).prepare().solver
    before = _configuration_fingerprint(solver)
    solver.Ex.fill_(2.)
    solver._update_nonlinear_electric_coefficients()
    assert _configuration_fingerprint(solver) == before


def test_resume_identity_covers_network_dynamics():
    from tests.rf.network.test_network_delay_checkpoint import _simulation as network_simulation

    solver = network_simulation().prepare().solver
    before = _configuration_fingerprint(solver)
    solver._network_runtimes[0].D.mul_(0.5)
    assert _configuration_fingerprint(solver) != before
