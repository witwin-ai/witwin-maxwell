"""Surface extinction work independently checks two-plane modal interference."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.runtime import SystemDecodeHandle
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.periodic_diffraction import rayleigh_power_balance
from witwin.maxwell.mom.incident_work import incident_surface_work

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_surface_work_matches_rayleigh_interference_and_gradient():
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    direction = (.6, 0., .8)
    sources = [mw.PlaneWave(direction=direction, polarization=(0., 1., 0.)),
        mw.PlaneWave(direction=(.6, 0., -.8), polarization=(0., 1., 0.),
                     source_time=mw.CW(frequency=3e8, amplitude=.4))]
    handle = SystemDecodeHandle(original.system, original.exterior, sources,
        mw.MoMConfig(integration=original.integration), {})
    sample = original.system.bodies[0].mesh.vertices
    lattice = LatticeSpec(sample.new_tensor([[1., 0., 0.], [0., 1., 0.]]),
                          sample.new_tensor([.6*original.exterior.k.real, 0., 0.]))
    torch.manual_seed(31)
    current = torch.randn(2*original.system.total_edges, device='cuda', dtype=torch.complex128,
                          requires_grad=True)
    axis = torch.arange(-2, 3, device='cuda')
    orders = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    ledger = rayleigh_power_balance(handle, current, lattice=lattice, orders=orders,
        lower_plane=-.5, upper_plane=.8, quadrature_order=8)
    work = incident_surface_work(handle, current, quadrature_order=8)
    torch.testing.assert_close(work['extinction_work_W'], ledger['extinction_power_W'], rtol=1e-11, atol=1e-14)
    surface_grad, = torch.autograd.grad(work['extinction_work_W'], current, retain_graph=True)
    modal_grad, = torch.autograd.grad(ledger['extinction_power_W'], current)
    torch.testing.assert_close(surface_grad, modal_grad, rtol=1e-11, atol=1e-14)
    assert len(work['per_body_work_W']) == 2 and not work['independent_material_work_verified']
    print({'surface_work_W': float(work['extinction_work_W'].detach()),
           'modal_extinction_W': float(ledger['extinction_power_W'].detach())})


def test_original_exterior_work_matches_radiated_modes():
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    from witwin.maxwell.mom.ewald_nearfield import EwaldSystemBlocks
    from witwin.maxwell.mom.incident_work import exterior_surface_work
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    body = original.system.bodies[0]
    system = SurfaceSystem.create([(0, body.name, body.basis, body.interior)])
    handle = SystemDecodeHandle(system, original.exterior, original.sources,
        mw.MoMConfig(integration=original.integration), {})
    sample = body.mesh.vertices
    lattice = LatticeSpec(sample.new_tensor([[.7, 0., 0.], [0., .8, 0.]]), sample.new_zeros(3))
    kernel = EwaldHelmholtzKernel(lattice, 3.5, real_shells=3, reciprocal_shells=4, quadrature_order=64)
    from dataclasses import replace
    integration = replace(original.integration, order=6, singular_order=6)
    provider = EwaldSystemBlocks(system, original.exterior, integration, kernel)
    torch.manual_seed(19)
    current = torch.randn(2*system.total_edges, device='cuda', dtype=torch.complex128)
    current[system.total_edges:] *= abs(original.exterior.eta)
    current.requires_grad_()
    axis = torch.arange(-3, 4, device='cuda')
    orders = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    modal = rayleigh_power_balance(handle, current, lattice=lattice, orders=orders,
        lower_plane=-.5, upper_plane=.5, quadrature_order=8)['scattered_power_W']
    work = exterior_surface_work(provider, current, edge_batch=4)['emitted_power_W']
    relative = float(((work-modal).abs()/modal.abs()).detach())
    print({'weak_emitted_W': float(work.detach()), 'modal_scattered_W': float(modal.detach()),
           'relative_difference': relative})
    torch.testing.assert_close(work, modal, rtol=1e-5, atol=1e-14)
    grad, = torch.autograd.grad(work, current, retain_graph=True)
    reference, = torch.autograd.grad(modal, current)
    torch.testing.assert_close(grad, reference, rtol=1e-4, atol=1e-12)
