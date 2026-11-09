"""Near-field PMCHWT sparse LU preconditioner acceptance."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.operator import IntegrationConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('backend', ['dense', 'pfft'])
def test_wavelength_sphere_answer_and_iterations(backend):
    """Sparse LU preserves the answer and beats leaf-64 on a small eps=4 sphere.

    Sparse LU requires the optional GPU direct dependency and an exclusive
    device reservation; the caller supplies the device locks before pytest.
    """
    frequency = 299792458.
    mesh = sphere_mesh(radius=.5, frequency=2, device='cuda')
    scene = mw.Scene(structures=[mw.Structure(
        mw.Mesh(mesh.vertices, mesh.faces, recenter=False, fill_mode='solid'), mw.Material(eps_r=4.))],
        sources=[mw.PlaneWave()])
    pfft = mw.PFFTConfig(order=3, grid_precision='complex128',
                         solve_dual_tolerance=1e-6, original_sample_tolerance=3e-2)

    def run(factory):
        extra = {'pfft': pfft} if backend == 'pfft' else {}
        config = mw.MoMConfig(operator_backend=backend, integration=IntegrationConfig(order=4, singular_order=4),
            riesz_storage='compact_rwg_v1' if backend == 'dense' else 'local_rwg_v1',
            stopping=StoppingRule(method='gmres', preconditioner='none', rtol=1e-9, atol=0.,
                                  maxiter=600, restart=300, stagnation_window=300),
            right_preconditioner=factory, **extra)
        prepared = mw.Simulation.mom(scene, frequency=frequency, config=config).prepare()
        result = prepared.run()
        x = result.surface_currents.scattered_coefficients
        residual = (prepared.solver.rhs-prepared.solver.handle.apply(x)).norm()/prepared.solver.rhs.norm()
        assert result.stats()['converged']
        assert float(residual) <= 1e-6
        return x, result.stats()

    plain, _ = run(None)
    leaf, leaf_stats = run(mw.LeafBlockRightPreconditioner(64))
    x, stats = run(mw.NearFieldRightPreconditioner(pfft=pfft))
    assert float((x-plain).norm()/plain.norm()) <= 1e-6
    assert float((x-leaf).norm()/leaf.norm()) <= 1e-6
    assert stats['iterations'] <= leaf_stats['iterations']
