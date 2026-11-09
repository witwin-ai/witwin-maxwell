"""Reproduce the documented 64-face periodic failure's weak-energy decomposition.

This diagnostic consumes an existing current snapshot; it never solves a system.
The fixture geometry/materials are intentionally fixed to the historical control.
"""
import argparse
import json
from dataclasses import replace
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.runtime import SystemDecodeHandle
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.periodic_diffraction import rayleigh_power_balance
from witwin.maxwell.mom.nearfield import scalar_regional_block
from witwin.maxwell.mom.lifting import build_lifted_rhs
from witwin.maxwell.mom.incident_work import incident_surface_work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', help='Trusted local gsp07-power-refinement-v2.pt artifact')
    parser.add_argument('--order', type=int, default=4)
    parser.add_argument('--singular-order', type=int, default=6)
    parser.add_argument('--enriched-residual', action='store_true',
                        help='Measure three refined-space rows of the exactly transferred saved current')
    args = parser.parse_args()
    saved = torch.load(args.snapshot, map_location='cuda', weights_only=False)
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                            device='cuda', dtype=torch.float64)*.07
    volume = TetrahedralVolumeMesh(vertices, torch.tensor([[0, 1, 2, 3]], device='cuda'), region_id=1).refine(2)
    basis = RWGBasis(SurfaceMesh(volume.boundary.vertices.clone(), volume.boundary.faces.clone()))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(0, 'body0', basis, interior)])
    baseline = IntegrationConfig(order=4, singular_order=6)
    handle = SystemDecodeHandle(system, exterior, [mw.PlaneWave()], mw.MoMConfig(integration=baseline), {})
    coefficients = saved['coefficients']
    if len(basis.mesh.faces) != 64 or coefficients.shape != (2*system.total_edges,):
        raise ValueError('Snapshot does not match the documented 64-face fixture')
    lattice = LatticeSpec(vertices.new_tensor([[.7, 0., 0.], [0., .8, 0.]]), vertices.new_zeros(3))
    ledger = rayleigh_power_balance(handle, coefficients, lattice=lattice,
        orders=saved['ledger']['lower']['orders'], lower_plane=-.5, upper_plane=.5, quadrature_order=6)
    for key in ('scattered_power_W', 'extinction_power_W', 'lossless_balance_relative_defect'):
        torch.testing.assert_close(ledger[key], saved['ledger'][key], rtol=1e-12, atol=1e-20)
    integration = replace(baseline, order=args.order, singular_order=args.singular_order)
    handle.integration = handle.body_handles[0].integration = integration
    if args.enriched_residual:
        from witwin.maxwell.mom.rwg_refinement import RWGRefinement
        from witwin.maxwell.mom.lifting import build_selected_lifted_rhs
        from witwin.maxwell.mom.ewald_nearfield import EwaldSystemBlocks
        from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
        transfer = RWGRefinement(basis)
        n = basis.n_edges
        fine_current = torch.cat((transfer.apply(coefficients[:n]), transfer.apply(coefficients[n:])))
        fine = transfer.fine
        fine_system = SurfaceSystem.create([(0, 'body0', fine, interior)])
        fine_handle = SystemDecodeHandle(fine_system, exterior, [mw.PlaneWave()],
            mw.MoMConfig(integration=integration), {})
        selected = torch.tensor([0, fine.n_edges//3, 2*fine.n_edges//3], device='cuda')
        rhs, receipt = build_selected_lifted_rhs(fine_handle.body_handles[0], selected)
        kernel = EwaldHelmholtzKernel(lattice, 3.5, real_shells=3, reciprocal_shells=8, quadrature_order=64)
        provider = EwaldSystemBlocks(fine_system, exterior, integration, kernel)
        action = provider.sampled_action(selected, fine_current, source_batch=fine.n_edges)
        scale = rhs.real.new_ones(len(rhs))
        scale[:len(selected)] = abs(exterior.eta)
        residual, reference = (action-rhs)*scale, rhs*scale
        print(json.dumps({'coarse_faces': len(basis.mesh.faces), 'fine_faces': len(fine.mesh.faces),
            'selected_edges': selected.tolist(),
            'balanced_sample_relative_residual': float(residual.norm()/reference.norm()),
            'balanced_sample_absolute_residual': float(residual.norm()),
            'reference_norm': float(reference.norm()), 'rhs_receipt': receipt,
            'global_residual_certified': False, 'new_solve': False, 'physical_qualification': False}, indent=2))
        return
    n = system.total_edges
    edges = torch.arange(n, device='cuda')
    matrix, _ = scalar_regional_block(basis, basis, edges, edges, (interior,), integration)
    rhs, _ = build_lifted_rhs(handle.body_handles[0])
    def work(value):
        return .5*(coefficients[:n].conj()*value[n:]-coefficients[n:].conj()*value[:n]).sum().real
    inner, forcing = work(matrix@coefficients), work(rhs)
    extinction = incident_surface_work(handle, coefficients, quadrature_order=12)['extinction_work_W']
    scatter = ledger['scattered_power_W']
    print(json.dumps({'order': args.order, 'singular_order': args.singular_order,
        'interior_equivalent_work_W': float(inner), 'rhs_work_W': float(forcing),
        'equation_inferred_exterior_work_W': float(forcing-inner),
        'modal_scattered_W': float(scatter), 'incident_extinction_W': float(extinction),
        'equation_power_residual_W': float(forcing-inner-scatter),
        'lossless_defect_W': float(extinction-scatter), 'new_solve': False,
        'physical_qualification': False}, indent=2))


if __name__ == '__main__':
    main()
