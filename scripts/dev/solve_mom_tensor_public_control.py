"""Registered actual anisotropic public solve; no physical qualification claim."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import traceback
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.dyadic_green import SpectralModeCache


def build_control_scene():
    """Reconstruct the fixed physical control without assembling an operator."""
    v = torch.tensor([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]],
                     device='cuda', dtype=torch.float64)*.04
    f = torch.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], device='cuda')
    triangles = v[f]
    normals = torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    f = torch.where(((normals*triangles.mean(1)).sum(-1) < 0)[:, None], f[:, [0, 2, 1]], f)
    frequency = 3e8
    material = mw.Material(epsilon_tensor=mw.Tensor3x3(((2., .1, 0.), (.1, 2.4, 0.), (0., 0., 3.))),
                           sigma_e=.8*2*math.pi*frequency*EPSILON_0)
    scene = mw.Scene(structures=[mw.Structure(mw.Mesh(v, f, recenter=False, fill_mode='solid', device='cuda'),
                                            material, name='anisotropic_body')], sources=[mw.PlaneWave()])
    return scene, frequency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--periodic-dimension',type=int,choices=(1,2),default=None,
        help='Use the scalar Ewald exterior with the same anisotropic interior')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root/'witwin/maxwell/mom').glob('*.py'))
    sources += [root/'witwin/maxwell'/name for name in ('media.py', 'simulation.py', 'result.py', '__init__.py')]
    def hashes():
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    start = time.monotonic()
    receipt = dict(source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), source_sha256=hashes(),
        status='running', physical_qualification=False, scope='actual_anisotropic_public_algebra_and_exact_restore',
        geometry='fixed_4_face_6_edge_tetrahedron', spectral_relative_tolerance=1e-4,
        spectral_absolute_tolerance=1e-9, max_radial_order=512, max_azimuthal_order=64,
        surface_order=1, singular_order=1, per_body_cache_budget_bytes=1024**3)
    receipt['periodic_dimension']=args.periodic_dimension
    def stage(name, **details):
        torch.cuda.synchronize()
        print(json.dumps(dict(stage=name, elapsed_seconds=time.monotonic()-start, **details)), flush=True)
    # Observational wrappers preserve every numerical cache operation and value.
    original_get = SpectralModeCache.get
    def observed(self, key, build):
        before = self.misses
        value = original_get(self, key, build)
        if self.misses != before and self.misses % 10 == 0:
            stage('spectral_cache_progress', cache_kind=type(self).__name__, stats=self.stats())
        return value
    SpectralModeCache.get = observed
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2))
    try:
        torch.cuda.reset_peak_memory_stats()
        scene, frequency = build_control_scene()
        config = mw.MoMConfig(subdivisions=0, riesz_storage='compact_rwg_v1',
            integration=IntegrationConfig(order=1, singular_order=1, pair_batch=4),
            dyadic_cache_max_bytes=1024**3,
            dyadic_integration=mw.DyadicIntegrationConfig(radial_order=8, azimuthal_order=8,
                max_radial_order=512, max_azimuthal_order=64, relative_tolerance=1e-4, absolute_tolerance=1e-9),
            query_config=mw.FieldQueryConfig(adaptive=False))
        if args.periodic_dimension is not None:
            config=replace(config,periodic=mw.MoMPeriodicConfig(
                vectors_m=((2.,0.,0.),(0.,2.,0.))[:args.periodic_dimension],
                real_shells=3,reciprocal_shells=4,quadrature_order=128))
            receipt['periodic_config']=config.periodic.payload()
            receipt['scope']='actual_periodic_tensor_public_algebra_and_exact_restore'
            (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2))
        stage('geometry_ready')
        prepared = mw.Simulation.mom(scene, frequency=frequency, config=config).prepare()
        stage('operator_and_rhs_ready')
        result = prepared.run()
        data, handle = result.surface_currents, prepared.solver.handle
        current, rhs = data.scattered_coefficients, prepared.solver.rhs
        residual = handle.apply(current)-rhs
        eta = abs(handle.exterior.eta)
        weight = current.real.new_tensor([eta]*6+[1.]*6)
        relative = float((residual*weight).norm()/(rhs*weight).norm())
        local = handle.body_handles[0]
        retained = local.tensor_kernel.mode_cache.retained_bytes+local.tensor_contrast_kernel.mode_cache.retained_bytes
        result.save(args.output/'result.pt')
        torch.save(dict(matrix=handle.matrix.detach().cpu(), rhs=rhs.detach().cpu(),
            solver=result.stats(), metadata=handle.metadata,
            regional_cache=local.tensor_kernel.mode_cache.stats(), contrast_cache=local.tensor_contrast_kernel.mode_cache.stats()),
            args.output/'diagnostics.pt')
        receipt.update(solver_converged=bool(result.stats()['converged']),
            full_original_balanced_residual=relative, retained_spectral_cache_bytes=retained,
            result_sha256=hashlib.sha256((args.output/'result.pt').read_bytes()).hexdigest())
        assert result.stats()['converged'] and math.isfinite(relative) and relative < 1e-8, receipt
        assert retained <= config.dyadic_cache_max_bytes
        restored = mw.Result.load(args.output/'result.pt', scene=scene)
        assert torch.equal(restored.surface_currents.scattered_coefficients, current)
        points = torch.tensor([[.3, .2, .4]], device='cuda', dtype=torch.float64)
        for actual, expected in zip(restored.query_fields(points, region_id='exterior', field_kind='scattered'),
                                    result.query_fields(points, region_id='exterior', field_kind='scattered')):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert hashes() == receipt['source_sha256'], 'Numerical source changed during registered control'
        receipt.update(status='measured', exact_restore=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        stage('control_complete', balanced_residual=relative)
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        torch.save(dict(error_type=type(error).__name__, message=str(error), receipt=getattr(error, 'receipt', None)),
                   args.output/'failure.pt')
        traceback.print_exc()
        raise
    finally:
        SpectralModeCache.get = original_get
        receipt['elapsed_seconds'] = time.monotonic()-start
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
