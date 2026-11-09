"""Solve registered planar refinements of the fixed periodic tetrahedron.

The 64-face solution has a stable enriched-test-space residual. This runner
solves its 256/1024-face successors through the public Ewald FMM runtime and retains
both numerical admission and physical power results, including failures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.periodic_diffraction import rayleigh_power_balance
from witwin.maxwell.mom.incident_work import incident_surface_work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--maxiter', type=int, default=500)
    parser.add_argument('--restart', type=int, default=120)
    parser.add_argument('--stagnation-window', type=int, default=20)
    parser.add_argument('--leaf-factor-kernel', choices=('none', 'original', 'primary'), default='none')
    parser.add_argument('--surface-refinements', type=int, choices=(1, 2), default=1)
    parser.add_argument('--table-storage', choices=('compact', 'dense'), default='compact')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    receipt = {'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ('witwin/maxwell/mom/ewald_nearfield.py', 'witwin/maxwell/mom/nearfield.py')},
        'leaf_factor_kernel': args.leaf_factor_kernel, 'table_storage': args.table_storage,
        'physical_qualification': False, 'new_faces': 64*4**args.surface_refinements,
        'surface_refinements': args.surface_refinements,
        'previous_faces': 64*4**(args.surface_refinements-1),
        'previous_lossless_defect': (.4904611857861883 if args.surface_refinements == 1
                                     else .07586217288114645),
        'motivation': 'unresolved_lossless_balance_under_planar_mesh_refinement',
        'iteration_budget': {'maxiter': args.maxiter, 'restart': args.restart,
                             'stagnation_window': args.stagnation_window}}
    def stage(name):
        torch.cuda.synchronize()
        print(json.dumps({'stage': name, 'elapsed_seconds': time.monotonic()-start}), flush=True)
    try:
        torch.cuda.reset_peak_memory_stats()
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                device='cuda', dtype=torch.float64)*.07
        volume = TetrahedralVolumeMesh(vertices, torch.tensor([[0, 1, 2, 3]], device='cuda'), region_id=1).refine(2)
        mesh = SurfaceMesh(volume.boundary.vertices.clone(), volume.boundary.faces.clone())
        for _ in range(args.surface_refinements):
            mesh = mesh.refine()
        if len(mesh.faces) != receipt['new_faces']:
            raise RuntimeError('Registered refinement face count changed')
        scene = mw.Scene(structures=[mw.Structure(mw.Mesh(mesh.vertices, mesh.faces,
            recenter=False, fill_mode='solid', device='cuda'), mw.Material(eps_r=2.), name='body0')],
            sources=[mw.PlaneWave()])
        config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=4, singular_order=6),
            periodic=mw.MoMPeriodicConfig(((.7, 0., 0.), (0., .8, 0.)), eta_per_m=3.5,
                real_shells=3, reciprocal_shells=8, quadrature_order=64),
            operator_backend='fmm', riesz_storage='local_rwg_v1',
            fmm=mw.FMMConfig(order=8, table_storage=args.table_storage, separation_ratio=.25,
                            leaf_factor_kernel=args.leaf_factor_kernel),
            stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal',
                rtol=1e-12, atol=0., maxiter=args.maxiter, restart=args.restart,
                stagnation_window=args.stagnation_window),
            query_config=FieldQueryConfig(adaptive=False))
        stage('geometry_ready')
        prepared = mw.Simulation.mom(scene, frequency=3e8, config=config).prepare()
        stage('fmm_and_rhs_ready')
        result = prepared.run()
        stage('solve_finished')
        result.save(args.output/'result.pt')
        data = result.surface_currents
        handle, coefficients = data.handle, data.scattered_coefficients
        axis = torch.arange(-3, 4, device='cuda')
        orders = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
        ledger = rayleigh_power_balance(handle, coefficients, lattice=handle.periodic_kernel.lattice,
            orders=orders, lower_plane=-.5, upper_plane=.5, quadrature_order=6)
        work = incident_surface_work(handle, coefficients, quadrature_order=12)
        receipt.update(status='measured', solver_converged=bool(result.stats()['converged']),
            solver_reason=result.stats().get('reason'),
            energy={key: float(ledger[key]) for key in ('scattered_power_W', 'extinction_power_W',
                'flux_inferred_absorption_W', 'lossless_balance_relative_defect')},
            incident_surface_work_W=float(work['extinction_work_W']),
            elapsed_seconds=time.monotonic()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            memory_scope='single_run_including_geometry_setup_solve_and_observables')
        torch.save({'ledger': ledger, 'incident_work': work, 'solver': result.stats(),
                    'construction': handle.actions.receipt()}, args.output/'diagnostics.pt')
        torch.testing.assert_close(work['extinction_work_W'], ledger['extinction_power_W'], rtol=1e-8, atol=1e-20)
        stage('power_measured')
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.monotonic()-start)
        traceback.print_exc()
        raise
    finally:
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
