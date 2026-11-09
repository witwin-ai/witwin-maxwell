"""Negative controls for MoM capability guards added after the guard-census baseline."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.host_validation
def test_lossy_metal_body_kind_rejects_surface_impedance_conductors():
    from witwin.maxwell.mom.topology import body_kind
    with pytest.raises(NotImplementedError, match='Leontovich'):
        body_kind(mw.Structure(mw.Sphere(radius=0.1), mw.LossyMetalMedium(conductivity=5.8e7)))


@pytest.mark.host_validation
def test_right_preconditioner_rejects_tensor_and_periodic_compositions():
    from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
    with pytest.raises(NotImplementedError, match='free-space scalar linear runtime'):
        mw.MoMConfig(stopping=StoppingRule(method='gmres'), right_preconditioner=mw.IdentityRightPreconditioner(),
                     dyadic_integration=DyadicIntegrationConfig())


@cuda
def test_trainable_sheet_geometry_is_rejected_before_assembly():
    position = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    scene = mw.Scene(subpixel_samples=mw.SubpixelSpec(samples=2, averaging="polarized"),
                     structures=[mw.Structure(mw.Box(position=position, size=(0.3, 0.2, 0)), mw.Material.pec())],
                     sources=[mw.PlaneWave(direction=(0., 0., -1.), polarization=(1., 0., 0.),
                                           source_time=mw.CW(frequency=3e8))], device='cuda')
    config = mw.MoMConfig(open_surface=mw.OpenSurfaceConfig(max_edge_length_m=0.1))
    with pytest.raises(NotImplementedError, match='derivatives are not supported'):
        mw.Simulation.mom(scene, frequency=3e8, config=config).prepare()


@cuda
def test_batched_lifted_rhs_rejects_periodic_lattices():
    from witwin.maxwell.mom.fields import FieldQueryConfig
    from witwin.maxwell.mom.operator import IntegrationConfig
    source = mw.PlaneWave()
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.07), mw.Material(eps_r=2.))], sources=[source])
    periodic = mw.MoMPeriodicConfig(vectors_m=((1., 0., 0.),), reciprocal_shells=4, quadrature_order=128)
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=2, singular_order=3),
                          query_config=FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1',
                          periodic=periodic)
    prepared = mw.Simulation.mom(scene, frequency=3e8, config=config).prepare()
    with pytest.raises(NotImplementedError, match='periodic lattices'):
        prepared.solver.handle.build_rhs_batch([[source]])
