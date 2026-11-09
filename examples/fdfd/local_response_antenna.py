"""Small nonuniform-grid, multifrequency native-GPU binary antenna design.

Run on physical GPU 1:
  CUDA_VISIBLE_DEVICES=1 conda run -n maxwell python examples/fdfd/local_response_antenna.py --passes 1

This is a compact reproducible workflow, not a reproduction of a published device.
"""
from __future__ import annotations
import argparse
import json
import torch
import witwin.maxwell as mw


def build(passes=1):
    lateral = [-.2,-.15,-.12,-.08,-.06,-.04,-.02,0.,.02,.04,.06,.08,.12,.15,.2]
    vertical = [-.2,-.15,-.12,-.08,-.04,-.02,0.,.02,.04,.08,.12,.15,.2]
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.2,.2),)*3),
                     grid=mw.GridSpec.custom(lateral,lateral,vertical),
                     boundary=mw.BoundarySpec.pml(num_layers=3,strength=4),device='cuda')
    scene.add_structure(mw.Structure(name='substrate',geometry=mw.Box(position=(0.,0.,-.01),size=(.16,.16,.02)),material=mw.Material(eps_r=3.5)))
    scene.add_structure(mw.Structure(name='ground',geometry=mw.Box(position=(0.,0.,-.02),size=(.16,.16,0.)),material=mw.Material(pec=True)))
    source = mw.LumpedCurrentSource(start=(-.02,0.,0.),end=(.02,0.,0.),current=1.)
    scene.add_source(source)
    observations = (mw.FieldProjection.voltage('feed',source.start,source.end),
                    mw.FarFieldProjection('far_x',bounds=((-0.12,.12),)*3,direction=(0.,0.,1.),polarization=(1.,0.,0.)),
                    mw.FarFieldProjection('far_y',bounds=((-0.12,.12),)*3,direction=(0.,0.,1.),polarization=(0.,1.,0.)))
    config = mw.LocalResponseConfig(mw.LocalDesignRegion(bounds=((-0.06,.06),(-0.06,.06),(0.,0.)),components=('Ex','Ey')),
                                   observations,rhs_batch_size=16,max_response_bytes=2**30,tolerance=1e-7)
    simulation = mw.Simulation.fdfd(scene,frequencies=(.9e9,1.e9,1.1e9),
                    solver=mw.GMRES(solver_type='direct',precision='double'),local_response=config)
    return simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--passes',type=int,default=1)
    parser.add_argument('--cache',help='Directory for reusable per-frequency background responses.')
    parser.add_argument('--reuse-cache',action='store_true')
    args = parser.parse_args()
    if args.reuse_cache and not args.cache:
        parser.error('--reuse-cache requires --cache')
    with build().prepare(response_cache=args.cache if args.reuse_cache else None) as sweep:
        if args.cache and not args.reuse_cache:
            sweep.save_cache(args.cache)
        base = sweep.responses[0]
        coordinates = []
        for component in ('Ex','Ey','Ez'):
            axes = [getattr(base.scene,a+('_half' if a==component[1].lower() else '')) for a in 'xyz']
            coordinates.append(torch.stack(torch.meshgrid(*axes,indexing='ij'),dim=-1).reshape(-1,3))
        points = torch.cat(coordinates)[base.design_indices]
        groups = []
        for x in (-.04,0.,.04):
            for y in (-.04,0.,.04):
                selected = ((points[:,0]-x).abs() <= .020001)&((points[:,1]-y).abs() <= .020001)
                # Keep the prescribed feed gap open during metal search.
                selected &= ~((points[:,1].abs()<1e-6)&(points[:,0].abs()<.020001))
                group = selected.nonzero().flatten()
                if group.numel():
                    groups.append(group)
        def objective(values):
            impedance = values[:,0]
            reflection = (impedance-50)/(impedance+50)
            accepted_power = .5*impedance.real
            eta = 4*torch.pi*1e-7*299792458.
            intensity = values[:,1:].abs().square().sum(dim=1)/(2*eta)
            # The example materials are lossless: gain equals directivity.
            directivity = 4*torch.pi*intensity/accepted_power.clamp_min(1e-12)
            return (200*reflection.abs().square()+(15-directivity).square()).mean()
        optimized = sweep.optimize_binary(groups,objective,max_passes=args.passes,seed=7)
        print(json.dumps({'frequencies':sweep.simulation.frequencies,
                          'design_dofs':base.design_indices.numel(),'tiles':optimized.tiles.cpu().tolist(),
                          'attempts':optimized.attempts,'accepted':optimized.accepted,'converged':optimized.converged,
                          'objective_history':optimized.objective_history.cpu().tolist(),
                          'cache_hits':[r.cache_hit for r in sweep.responses],
                          'precompute_seconds':[r.precompute_seconds for r in sweep.responses]},indent=2))


if __name__ == '__main__':
    main()
