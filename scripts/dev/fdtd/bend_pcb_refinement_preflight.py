"""CPU-only geometry and source-duality preflight for bent-board refinement."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.compiler.ports import compile_port_geometry
from witwin.maxwell.scene import PreparedScene
from scripts.dev.fdtd.bend_material_operator_audit import native_scene
from scripts.dev.fdtd.bend_pcb_external_reference import fixture
from scripts.dev.fdtd.matched_current_reference import compile_known_current_edges
from scripts.dev.fdtd.pcb_acceptance import package_identity


def compile_source_duality(spacing):
    """Compile CPU geometry only; artifact qualification is handled by run()."""
    cuda_before=torch.cuda.is_initialized()
    model=fixture(spacing,3.,grid_construction='domain_then_pml',canonical_grid_faces=True)
    radial,angular,axial=model['grid']
    lengths=(np.diff(radial).min(),radial[0]*np.diff(angular).min(),np.diff(axial).min())
    dt=.4/(299792458.*np.sqrt(sum(1/value**2 for value in lengths)))
    lower,upper=model['aperture']
    edges=compile_known_current_edges(model['grid'],axis=0,path=(upper[0],lower[0]),
        aperture_bounds={1:(lower[1],upper[1]),2:(lower[2],upper[2])},dt=dt,
        epsilon=2.2*8.8541878128e-12,conductivity=.001,cylindrical_radial=True)
    port=mw.LumpedPort('feed',positive=(.01,.02,0.),negative=(.01,0.,0.),voltage_path=mw.AxisPath('y'),
        reference_impedance=1e12,current_surface=mw.Box(position=torch.tensor((.01,spacing/2,0.),dtype=torch.float64),
        size=torch.tensor((.02+spacing,0.,.02+spacing),dtype=torch.float64)),
        voltage_aperture=mw.Box(position=torch.tensor((.01,.01,0.),dtype=torch.float64),
        size=torch.tensor((.02,.02,.02),dtype=torch.float64)))
    scene=PreparedScene(native_scene(spacing).clone(ports=(port,)))
    compiled=compile_port_geometry(scene,port,device='cpu')
    native={tuple(index):float(weight)for index,weight in zip(compiled.voltage_indices.tolist(),compiled.voltage_weights.tolist())}
    expected={(edge.index[1],len(radial)-2-edge.index[0],edge.index[2]):
        -edge.current_density_per_amp*edge.dual_area*edge.edge_length for edge in edges}
    missing=sorted(set(expected)-set(native));extra=sorted(set(native)-set(expected))
    common=set(expected)&set(native)
    relative=max((abs(native[index]-expected[index])/abs(expected[index])for index in common),default=math.inf)
    flux=[sum(edge.current_density_per_amp*edge.dual_area for edge in edges if edge.index[0]==index)
        for index in sorted({edge.index[0]for edge in edges})]
    if torch.cuda.is_initialized() and not cuda_before:raise RuntimeError('CPU source preflight initialized CUDA.')
    return dict(schema='bent-pcb-refinement-source-preflight/1',
        spacing_m=spacing,curvature_m_inverse=3.,metal='pec',duration_s=96e-9,source_peak_current_a=.001,
        independent_dt_s=dt,independent_steps=math.ceil(96e-9/dt),grid_points=[len(axis)for axis in model['grid']],
        cells=int(np.prod([len(axis)-1 for axis in model['grid']])),minimum_padded_metric=.46,
        independent_grid_sha256=hashlib.sha256(b''.join(axis.tobytes()for axis in model['grid'])).hexdigest(),
        independent_source_edges=len(edges),native_source_edges=len(native),missing_native_edges=missing,extra_native_edges=extra,
        maximum_relative_voltage_weight_error=relative,radial_current_flux_per_amp=flux,
        voltage_weight_sum=sum(native.values()),physical_gap_m=.02,face_roundoff=model['face_roundoff'],
        contract='Native positive Eu voltage weights equal minus physical radial current density per ampere times clipped dual-face area and edge length; Er=-Eu. No fitted normalization.',
        cpu_only=True,cuda_context_preexisting=cuda_before,cuda_context_initialized_by_preflight=False,
        source_peak_current_native_contract='Native high-R drive is measured spectrally; finite-R feedback must be bounded from the resulting impedance.',
        passed=not missing and not extra and relative<2e-6 and max(abs(abs(value)-1)for value in flux)<1e-12)


def run(output,spacing):
    root=Path(__file__).resolve().parents[3]
    if not (root/'SOURCE_BASE_COMMIT').exists() or not Path(mw.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Refinement preflight requires its frozen source archive.')
    paths=(Path(__file__),Path(native_scene.__code__.co_filename),Path(fixture.__code__.co_filename),
        Path(compile_known_current_edges.__code__.co_filename))
    def binding():return dict(base_commit=(root/'SOURCE_BASE_COMMIT').read_text().strip(),
        native_sha256=package_identity(mw,'witwin-maxwell')['source_sha256'],
        helpers={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in paths})
    before=binding();report=compile_source_duality(spacing);after=binding()
    if before!=after:raise RuntimeError('Preflight source changed during compilation.')
    report.update(source_binding_before=before,source_binding_after=after)
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spacing',type=float,default=.0025);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    if not run(args.output,args.spacing)['passed']:raise SystemExit('Refined physical source duality failed.')
