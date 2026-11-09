"""Measure actual tensor interior/trace recovery from an existing periodic root."""
import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import torch
import witwin.maxwell as mw
from solve_mom_tensor_public_control import build_control_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / 'witwin/maxwell/mom').glob('*.py'))
    files += [root / 'witwin/maxwell/result.py', Path(__file__),
              Path(__file__).with_name('solve_mom_tensor_public_control.py')]

    def hashes():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files}

    start = time.monotonic()
    receipt = dict(status='running', physical_qualification=False,
                   scope='saved_root_actual_tensor_interior_and_trace_quadrature',
                   result_path=str(args.result.resolve()),
                   result_sha256=hashlib.sha256(args.result.read_bytes()).hexdigest(),
                   source_before=hashes(), cases=[])

    def write():
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))

    def forbidden(*args, **kwargs):
        raise AssertionError('Saved field recovery must not assemble or solve a new root')

    write()
    try:
        producer_path = args.result.with_name('receipt.json')
        producer_bytes = producer_path.read_bytes()
        producer = json.loads(producer_bytes)
        receipt['producer_receipt_path'] = str(producer_path.resolve())
        receipt['producer_receipt_sha256'] = hashlib.sha256(producer_bytes).hexdigest()
        assert producer['status'] == 'measured', 'Original root control did not pass'
        assert producer['scope'] == 'actual_periodic_tensor_public_algebra_and_exact_restore'
        assert producer['solver_converged'] is True and producer['exact_restore'] is True
        residual = producer['full_original_balanced_residual']
        assert math.isfinite(residual) and 0 <= residual < 1e-8
        assert 0 <= producer['retained_spectral_cache_bytes'] <= producer['per_body_cache_budget_bytes']
        assert producer['result_sha256'] == receipt['result_sha256'], 'Producer/result hash mismatch'
        receipt['producer_balanced_residual'] = residual
        torch.use_deterministic_algorithms(True)
        scene, _ = build_control_scene()
        with ExitStack() as stack:
            for target in ('witwin.maxwell.mom.periodic_runtime.PeriodicSystemOperator.__init__',
                           'witwin.maxwell.mom.runtime.SystemOperator.__init__',
                           'witwin.maxwell.mom.anisotropic_operator.assemble_generalized_system',
                           'witwin.maxwell.mom.periodic_runtime.assemble_periodic_system'):
                stack.enter_context(patch(target, forbidden))
            result = mw.Result.load(args.result, scene=scene)
            data = result.surface_currents
            assert data.config.periodic is not None and data.config.dyadic_integration is not None
            periodic = data.config.periodic
            assert producer['periodic_config'] == periodic.payload()
            receipt['periodic_config'] = periodic.payload()
            receipt['nonzero_bloch_phase_control'] = any(periodic.bloch_rad_per_m)
            config = mw.FieldQueryConfig(relative_tolerance=1e-4,
                                        absolute_tolerance=1e-9, max_order=8)
            receipt['field_policy'] = dict(relative_tolerance=1e-4,
                                           absolute_tolerance=1e-9, max_order=8)
            basis = data.handle.body_handles[0].basis
            interior = basis.mesh.vertices.new_tensor([[.002, .003, .004]])
            trace = basis.mesh.triangles[:1].mean(1)
            translation = interior.new_tensor(periodic.vectors_m[0])
            phase = torch.exp(1j * (translation @ interior.new_tensor(periodic.bloch_rad_per_m)))
            for name, point, side in (('interior', interior, None),
                                      ('interior_trace', trace, 'interior')):
                case = dict(name=name, status='running')
                receipt['cases'].append(case)
                write()
                points = torch.cat((point, point + translation), dim=0)
                try:
                    fields = result.query_fields(points, region_id=1, field_kind='total',
                                                 trace_side=side, query_config=config)
                    log = copy.deepcopy(data.handle.last_periodic_field_log)
                    torch.save(dict(points=points.detach().cpu(),
                                    E=fields[0].detach().cpu(), H=fields[1].detach().cpu(),
                                    field_receipt=log), args.output / (name + '.pt'))
                    assert log['converged'] and log['adaptive']
                    assert log['field_error_metric'] == 'separate_E_and_vacuum_impedance_H'
                    for field in fields:
                        assert bool(torch.isfinite(field).all())
                        torch.testing.assert_close(field[1], phase * field[0], rtol=1e-10, atol=1e-12)
                    case.update(status='passed', field_receipt=log)
                except Exception as error:
                    case.update(status='failed', error_type=type(error).__name__,
                                error=str(error))
                    torch.save(dict(error_type=type(error).__name__, message=str(error),
                                    receipt=getattr(error, 'receipt', None)),
                               args.output / (name + '-failure.pt'))
                    traceback.print_exc()
                write()
                print(json.dumps(case), flush=True)
        receipt['source_after'] = hashes()
        assert receipt['source_before'] == receipt['source_after'], 'Control sources changed'
        assert receipt['result_sha256'] == hashlib.sha256(args.result.read_bytes()).hexdigest()
        assert producer_bytes == producer_path.read_bytes(), 'Producer receipt changed'
        receipt['status'] = 'passed' if all(c['status'] == 'passed' for c in receipt['cases']) else 'failed'
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - start
        write()
        print(json.dumps({k: v for k, v in receipt.items()
                          if k not in ('source_before', 'source_after')}), flush=True)
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
