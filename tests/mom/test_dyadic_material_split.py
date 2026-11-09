"""Complete-domain split mapping, independent scalar reference and cache identity."""
from dataclasses import asdict
import math
import pytest
import torch
from witwin.maxwell.mom import dyadic_green as dg
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.quadrature import gauss_rule
from witwin.maxwell.constants import EPSILON_0,MU_0

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def medium():
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    return TensorMedium(3e8,(2+.3j)*eye,(1+.2j)*eye)


def test_split_radial_complete_domain_moments_and_identity():
    nodes,weights=gauss_rule(128,device='cuda',dtype=torch.float64)
    rho,w=dg._weyl_radial_rule(nodes,weights,6.,.03,material_split=nodes.new_tensor(4.))
    assert len(rho)==256 and bool((w>0).all()) and bool(rho[-1]>1e4)
    for decay in (.2,1.,4.):
        actual=(w*torch.exp(-decay*rho)).sum()
        torch.testing.assert_close(actual,actual.new_tensor(1/decay**2),rtol=1e-12,atol=1e-13)
    config=dg.DyadicIntegrationConfig(radial_map='material_split_v1')
    assert asdict(config)['radial_map']=='material_split_v1'
    with pytest.raises(ValueError,match='radial mapping'):dg.DyadicIntegrationConfig(radial_map='cutoff')
    with pytest.raises(ValueError,match='single regional'):
        dg.integrate_spatial_green(nodes.new_tensor([[.1,.02,.03]]),medium(),radial_order=4,azimuthal_order=4,
            radial_map='material_split_v1',_exterior=medium())


def test_split_regular_kernel_matches_independent_scalar_formula():
    m=medium();point=m.epsilon_r.real.new_tensor([[.08,.02,.04]])
    actual=dg.integrate_spatial_green(point,m,radial_order=128,azimuthal_order=64,
                                     radial_map='material_split_v1',restore_static=False)
    expected,_=dg.isotropic_dyadic_green(point,m,regular_part=True)
    eta=math.sqrt(MU_0/EPSILON_0);scale=point.new_tensor([1.]*3+[eta]*3)
    error=torch.linalg.matrix_norm(scale[:,None]*(actual-expected)*scale[None,:])
    norm=torch.linalg.matrix_norm(scale[:,None]*expected*scale[None,:])
    assert bool((error<=1e-10+1e-5*norm).all())


def test_split_cache_isolation_and_adaptive_receipt():
    m=medium();point=m.epsilon_r.real.new_tensor([[.08,.02,.04]])
    cache=dg.SpectralModeCache(m)
    dg.integrate_spatial_green(point,m,radial_order=4,azimuthal_order=4,mode_cache=cache)
    split=dg.integrate_spatial_green(point,m,radial_order=4,azimuthal_order=4,
                                    radial_map='material_split_v1',mode_cache=cache)
    assert cache.misses==2 and cache.stats()['modal_matrices_built']==48
    repeat=dg.integrate_spatial_green(point,m,radial_order=4,azimuthal_order=4,
                                     radial_map='material_split_v1',mode_cache=cache)
    torch.testing.assert_close(repeat,split,rtol=0,atol=0)
    assert cache.hits==1
    config=dg.DyadicIntegrationConfig(32,16,256,128,radial_map='material_split_v1')
    _,receipt=dg.evaluate_spatial_green(point,m,config,regular_part=True)
    assert receipt['converged'] and receipt['radial_nodes_per_order']==2
    assert receipt['kernel_version']=='dyadic_weyl_residue_material_split_v1'
    assert receipt['finite_transverse_cutoff'] is None and receipt['artificial_loss']==0
    import witwin.maxwell as mw
    from witwin.maxwell.mom.runtime import _numerical_identity
    assert _numerical_identity(mw.MoMConfig(dyadic_integration=config))['dyadic_integration']['radial_map']=='material_split_v1'


@pytest.mark.parametrize('dimension', (1, 2, 3))
def test_split_periodic_canonical_payload(dimension):
    from dataclasses import replace
    from test_periodic_dyadic_regular import problem, QuasiperiodicDyadicRegularKernel
    regular = replace(problem(dimension), integration=dg.DyadicIntegrationConfig(radial_map='material_split_v1'))
    payload = regular.payload()
    restored = QuasiperiodicDyadicRegularKernel.from_payload(
        payload, medium=regular.medium, lattice=regular.lattice)
    assert restored.payload() == payload and restored.identity == regular.identity
    for invalid in ('infinite_rational', 'cutoff', 1, True):
        payload['integration']['radial_map'] = invalid
        with pytest.raises(ValueError, match='radial_map'):
            QuasiperiodicDyadicRegularKernel.from_payload(
                payload, medium=regular.medium, lattice=regular.lattice)
