"""Measured streamed-PMCHWT feasibility with actual CUDA solves and observables."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
import witwin.maxwell as mw
from benchmark.mom_matrix_free import StreamedWeakOperator, experimental_gmres
from benchmark.mom_fdtd import PROCESS_PROVENANCE, tetrahedral_volume_rule
from witwin.maxwell.mom.operator import assemble_weak, build_rhs, IntegrationConfig
from witwin.maxwell.mom.runtime import compile_surface, DecodeHandle
from witwin.maxwell.mom.fields import decode, far_field
from witwin.maxwell.mom.observables import volume_absorption


def relative(a,b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))


def measured(call):
    torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    result=call()
    torch.cuda.synchronize()
    return result,dict(seconds=time.perf_counter()-start,baseline_allocated_bytes=baseline,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-baseline)


def run_case(subdivision, order, *, solve):
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1),mw.Material(eps_r=2.25,sigma_e=.004))],
        sources=[mw.PlaneWave(direction=(0,0,1),polarization=(1,0,0),source_time=mw.CW(frequency=3e8,phase=.23))])
    integration=IntegrationConfig(order=order,singular_order=order)
    config=mw.MoMConfig(subdivisions=subdivision,integration=integration)
    basis,exterior,interior=compile_surface(scene,3e8,config)
    (matrix,metadata),dense_setup=measured(lambda:assemble_weak(basis,exterior,interior,integration))
    operator,setup=measured(lambda:StreamedWeakOperator(basis,exterior,interior,integration,pair_chunk=1024))
    rhs=build_rhs(basis,exterior,scene.sources)
    exact,dense_solve=measured(lambda:torch.linalg.solve(matrix,rhs))
    generator=torch.Generator(device='cuda').manual_seed(20260912)
    x=torch.randn((2,len(rhs)),device='cuda',dtype=matrix.dtype,generator=generator)
    y=torch.randn((2,len(rhs)),device='cuda',dtype=matrix.dtype,generator=generator)
    dense_forward,dense_forward_time=measured(lambda:x@matrix.mT)
    dense_adjoint,dense_adjoint_time=measured(lambda:y@matrix.conj())
    # Release the dense reference matrix before measuring the streamed actions.
    matrix_bytes=matrix.numel()*matrix.element_size()
    del matrix
    torch.cuda.empty_cache()
    forward,forward_time=measured(lambda:operator.apply(x));forward_log=dict(operator.last_action)
    adjoint,adjoint_time=measured(lambda:operator.adjoint(y));adjoint_log=dict(operator.last_action)
    physical_image,physical_time=measured(lambda:operator.apply(exact))
    row=dict(subdivision=subdivision,edges=basis.n_edges,frequency_Hz=3e8,
        conductivity_S_per_m=.004,epsilon_r=[interior.epsilon_r.real,interior.epsilon_r.imag],
        integration=asdict(integration),pair_chunk=1024,triangle_pair_counts=metadata['triangle_pair_counts'],
        dense_matrix_bytes=matrix_bytes,dense_setup=dense_setup,dense_solve=dense_solve,
        dense_forward=dense_forward_time,dense_adjoint=dense_adjoint_time,
        streamed_setup=setup,streamed_forward=forward_time,streamed_adjoint=adjoint_time,
        forward_integration_log=forward_log,adjoint_integration_log=adjoint_log,
        forward_relative_error=relative(forward,dense_forward),adjoint_relative_error=relative(adjoint,dense_adjoint),
        adjoint_dot_relative_error=float(abs((forward.conj()*y).sum()-(x.conj()*adjoint).sum())/abs((forward.conj()*y).sum())),
        dense_current_streamed_true_residual=relative(physical_image,rhs),physical_action_time=physical_time,
        dense_matrix_released_before_streamed_actions=True,
        repeated_integration_per_action=True,cross_action_numerical_cache=False,
        setup_scope='Geometry already compiled; streaming setup only retains basis/media/config references',
        timing_scope='CUDA synchronized single action; streamed action has two RHS; not isolated throughput')
    if solve:
        (actual,stats),solve_cost=measured(lambda:experimental_gmres(operator,rhs,rtol=1e-9))
        if not stats['converged']:
            raise RuntimeError(f'Streamed physical solve did not converge: {stats}')
        scale=torch.cat((torch.full((basis.n_edges,),abs(exterior.eta),device='cuda',dtype=torch.float64),torch.ones(basis.n_edges,device='cuda',dtype=torch.float64)))
        handle=DecodeHandle(basis,exterior,interior,tuple(scene.sources),integration,{})
        points=torch.tensor([[.105,0,0],[0,.105,0],[0,0,.105],[.14,.03,.02]],device='cuda',dtype=torch.float64)
        actual_e,actual_h=decode(handle,actual,points,'exterior','scattered')
        expected_e,expected_h=decode(handle,exact,points,'exterior','scattered')
        volume_points,volume_weights=tetrahedral_volume_rule(basis.mesh,degree=2)
        actual_power=float(volume_absorption(handle,actual,volume_points,volume_weights))
        expected_power=float(volume_absorption(handle,exact,volume_points,volume_weights))
        directions=torch.eye(3,device='cuda',dtype=torch.float64)
        actual_far=far_field(handle,actual,directions)[0]
        expected_far=far_field(handle,exact,directions)[0]
        row.update(streamed_solve=stats,streamed_solve_cost=solve_cost,
            current_relative_impedance_balanced_error=relative(actual*scale,exact*scale),
            near_query_points_m=points.cpu().tolist(),near_scattered_E_relative_error=relative(actual_e,expected_e),
            near_scattered_H_relative_error=relative(actual_h,expected_h),far_relative_error=relative(actual_far,expected_far),
            independent_volume_points=len(volume_points),streamed_volume_absorption_W=actual_power,
            dense_volume_absorption_W=expected_power,volume_absorption_relative_error=abs(actual_power/expected_power-1),
            observable_scope='Independent streamed-solve coefficients versus independently dense-solved coefficients on the identical discretization',
            solved_observations={'dense_current':{'real':exact.real.cpu().tolist(),'imag':exact.imag.cpu().tolist()},
                'streamed_current':{'real':actual.real.cpu().tolist(),'imag':actual.imag.cpu().tolist()},
                'dense_far':{'real':expected_far.real.cpu().tolist(),'imag':expected_far.imag.cpu().tolist()}})
    return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[1]
    source={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ['benchmark/mom_matrix_free.py','benchmark/mom_matrix_free_acceptance.py']}
    report=dict(scope='WP07 bounded matrix-free feasibility experiment, no public backend or fast-operator acceleration claim',
        complete_G0=False,source_sha256=source,provenance_segments=[PROCESS_PROVENANCE],
        physical_contract=dict(radius_m=.1,frequency_Hz=3e8,epsilon_real=2.25,conductivity_S_per_m=.004,
            mu_r=1,exterior='vacuum',incident_direction=[0,0,1],incident_polarization=[1,0,0],incident_amplitude=1,incident_phase_radians=.23),
        acceptance=dict(action_relative_budget=1e-11,streamed_true_residual_budget=1e-9,
            same_discretization_observable_relative_budget=1e-7),cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for subdivision,order,solve in [(0,4,True),(0,6,True),(1,4,False)]:
        row=run_case(subdivision,order,solve=solve)
        report['cases'].append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='solved_observations'}),flush=True)
    report['all_five_pair_classes_exercised']=all(sum(row['triangle_pair_counts'][category] for row in report['cases'])>0 for category in range(5))
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
