"""Compare leaf-block right preconditioning on a saved scalar periodic control."""
import argparse
import hashlib
import subprocess
import json
from pathlib import Path
import time
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.runtime import SurfaceData
from witwin.maxwell.mom.periodic_runtime import PeriodicMultipoleSystemOperator
from witwin.maxwell.mom.block_preconditioner import LeafBlockInverse, RightPreconditionedOperator
from witwin.maxwell.mom.linalg import StoppingRule, solve_reference
from witwin.maxwell.mom.periodic_diffraction import rayleigh_power_balance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved_result', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-block-edges', type=int, default=32)
    parser.add_argument('--factor-kernel', choices=('original', 'primary'), default='original')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    receipt = {'status': 'started', 'physical_qualification': False,
        'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'preconditioner_sha256': hashlib.sha256(Path('witwin/maxwell/mom/block_preconditioner.py').read_bytes()).hexdigest()}
    def stage(name):
        torch.cuda.synchronize()
        print(json.dumps({'stage': name, 'elapsed_seconds': time.monotonic()-start}), flush=True)
    try:
        saved = torch.load(args.saved_result, map_location='cpu', weights_only=False)['surface_data']
        if len(saved['body_table']) != 1:
            raise ValueError('This control runner requires the fixed one-body periodic fixture')
        body = saved['body_table'][0]
        scene = mw.Scene(structures=[mw.Structure(mw.Mesh(body['vertices'].to('cuda'), body['faces'].to('cuda'),
            recenter=False, fill_mode='solid', device='cuda'), mw.Material(eps_r=2.), name='body0')],
            sources=[mw.PlaneWave()])
        data = SurfaceData.restore(saved, scene, 3e8)
        base = PeriodicMultipoleSystemOperator(data.handle.system, data.handle.exterior, scene.sources,
            data.config, kernel=data.handle.periodic_kernel)
        rhs = base.build_rhs()
        stage('original_fmm_ready')
        inverse = LeafBlockInverse(base.provider, base.tree, max_block_edges=args.max_block_edges, factor_kernel=args.factor_kernel)
        view = RightPreconditionedOperator(base, inverse)
        stage('leaf_inverse_ready')
        y, solver = solve_reference(view, rhs, StoppingRule(method='gmres', preconditioner='rwg_diagonal',
            rtol=1e-12, atol=0., maxiter=500, restart=120, stagnation_window=20))
        current = inverse.apply(y)
        stage('krylov_finished')
        report = base.residual_report(current, rhs)
        admitted = bool(solver['converged']) and all(
            value is not None and value <= 1e-8 for key in ('compressed_dual', 'original_sampled')
            for value in report[key]['relative'])
        axis = torch.arange(-3, 4, device='cuda')
        orders = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
        ledger = rayleigh_power_balance(base, current, lattice=base.periodic_kernel.lattice,
            orders=orders, lower_plane=-.5, upper_plane=.5, quadrature_order=6)
        receipt.update(status='measured', numerical_admission=admitted, inverse=inverse.receipt(),
            iterations=solver['iterations'], solver_reason=solver['reason'], residual_report=report,
            lossless_balance_relative_defect=float(ledger['lossless_balance_relative_defect']),
            previous_current_relative_difference=float((current-data.scattered_coefficients).norm()/data.scattered_coefficients.norm()),
            elapsed_seconds=time.monotonic()-start)
        torch.save({'coefficients': current, 'solver': solver, 'residual_report': report,
                    'ledger': ledger, 'inverse': inverse.receipt()}, args.output/'diagnostics.pt')
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
