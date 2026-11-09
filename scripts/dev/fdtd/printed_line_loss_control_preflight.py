"""CPU-only material-control and fixed-functional mesh preflight; no field advance."""
from __future__ import annotations
import argparse, dataclasses, json, math, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase, DRIVE_BASIS, build_case, make_simulation, execution_identity, tensor_signature
from scripts.dev.fdtd.printed_line_broadband import encode
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.copper_foil import compile_copper_faces, compile_copper_response
from witwin.maxwell.compiler.ports import compile_ports
from witwin.maxwell.constants import C_0, EPSILON_0, MU_0
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.fdtd.runtime.timestep import compile_fdtd_time_step
from witwin.maxwell.fdtd.runtime.materials import _electric_update_coefficients
from witwin.maxwell.fdtd.lumped import prepare_lumped_runtime


def main():
    initial_cuda = torch.cuda.is_initialized()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metal',choices=('pec','copper'),required=True);parser.add_argument('--sigma',type=float,required=True)
    parser.add_argument('--dx',type=float,default=.0025);parser.add_argument('--longitudinal-dl',type=float,default=.0025)
    parser.add_argument('--record-legacy-native-vi',action='store_true')
    parser.add_argument('--probe-reference-dx',type=float);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    identity=execution_identity()
    case=PilotCase(metal=args.metal,sigma=args.sigma,dx=args.dx,longitudinal_dl=args.longitudinal_dl,probe_reference_dx=args.probe_reference_dx,record_legacy_native_vi=args.record_legacy_native_vi)
    scene,groups,bounds,schema,rows,expected,tails=build_case(case)
    prepared=prepare_scene(scene);model=prepared.compile_materials();edge=model['edge_components']
    simulation=make_simulation(scene,groups,DRIVE_BASIS[0],case,graph=True,steps=1,device='cpu')
    frequency=simulation._fdtd_time_step_frequency(prepared)
    context=SimpleNamespace(scene=prepared,c=C_0,eps0=EPSILON_0,mu0=MU_0,full_aniso_enabled=False,
        dt=FDTD.auto_dt(None,*[float(getattr(prepared,f'd{axis}_primal64').min()) for axis in 'xyz'],frequency),
        _requested_source_frequency=frequency)
    for name in ('Ex','Ey','Ez'):setattr(context,'eps_'+name,edge['eps'][name]*EPSILON_0)
    for name in ('Hx','Hy','Hz'):setattr(context,'mu_'+name,edge['mu'][name]*MU_0)
    compile_fdtd_time_step(context,model)
    dt=float(context.dt);count=math.ceil(case.duration/dt);split=math.floor(case.checkpoint_time/dt)
    shapes={name:list(edge['eps' if name.startswith('E') else 'mu'][name].shape) for name in ('Ex','Ey','Ez','Hx','Hy','Hz')}
    copper=[]
    for face in compile_copper_faces(prepared,device='cpu'):
        response=compile_copper_response(face.material)
        copper.append(dict(face=face.name,component='E'+'xyz'[face.component],normal_axis='xyz'[face.axis],
            actual_geometry_edge_count=len(face.indices),indices_sha256=tensor_signature(face.indices),
            rates=len(response.rates),B_shape=list(response.state_space.B.shape),D_shape=list(response.state_space.D.shape),
            constitutive_relative_error_bound=response.relative_error_bound))
    assert bool(copper)==(case.metal=='copper')
    ports=[]
    for port,geometry in zip(scene.ports,compile_ports(prepared,device='cpu')):
        component=geometry.voltage_component;eps=edge['eps'][component]*EPSILON_0;sigma=edge['sigma_e'][component]
        _,coefficient=_electric_update_coefficients(SimpleNamespace(dt=dt,conductive_enabled=case.sigma>0),eps,sigma,None)
        own='xyz'.index(component[1].lower())
        widths=[torch.as_tensor(getattr(prepared,f'd{axis}_{"primal" if dimension==own else "dual"}64'),dtype=eps.dtype) for dimension,axis in enumerate('xyz')]
        volume=widths[0][:,None,None]*widths[1][None,:,None]*widths[2][None,None,:]
        runtime=prepare_lumped_runtime(geometry,dt=dt,eps_edge=eps,electric_update_coefficient=coefficient,
            yee_control_volume=volume,resistance=50.)
        stamping={name:getattr(runtime,name) for name in ('linear_indices','voltage_weights','injection','dt','resistance',
            'coupling_impedance','discrete_port_impedance','inverse_denominator','parallel_total_admittance',
            'inductance','capacitance','series_inductive_impedance','series_capacitive_impedance','topology')}
        ports.append(dict(name=port.name,positive=port.positive,negative=port.negative,
            voltage_aperture=dict(position=port.voltage_aperture.position,size=port.voltage_aperture.size),
            diagnostic_current_surface=dict(position=port.current_surface.position,size=port.current_surface.size),
            voltage_indices_sha256=tensor_signature(geometry.voltage_indices),voltage_weights_sha256=tensor_signature(geometry.voltage_weights),
            voltage_row_weight_sum_m=float(geometry.voltage_weights.sum()),Norton_resistance_ohm=50.,
            CPU_native_Norton_stamp_sha256=tensor_signature(stamping),
            current_indices_sha256=tensor_signature(geometry.current_indices)))
    plane_bytes=sum(len(value['planes'])*math.prod(value['full_shape'][:2])*len(case.frequencies)*8 for value in schema.values())
    field_bytes=sum(math.prod(shape)*4 for shape in shapes.values())
    state_bytes=sum(value['actual_geometry_edge_count']*value['rates']*8 for value in copper)
    spectral_bytes=sum(value['actual_geometry_edge_count']*2*len(case.frequencies)*16 for value in copper)
    record=dict(schema='loss-material-control-CPU-preflight/1',created_unix_s=time.time(),case=dataclasses.asdict(case),
        CPU_compiled_time_step_forecast_s=dt,CPU_time_step_metadata=context._time_step_metadata,
        planned_steps_forecast=count,checkpoint_step_forecast=split,planned_duration_s=count*dt,
        grid_shape=[prepared.Nx,prepared.Ny,prepared.Nz],domain_bounds_m=scene.domain.bounds,ROI_bounds_m=bounds,
        grid_node_sha256={axis:tensor_signature(getattr(prepared,axis+'_nodes64')) for axis in 'xyz'},
        physical_source_ports=ports,source_basis=DRIVE_BASIS,source_group_names=groups,
        waveform=dataclasses.asdict(__import__('witwin.maxwell',fromlist=['GaussianPulse']).GaussianPulse(1e9,fwidth=.8e9)),
        actual_native_component_shapes=shapes,native_observation_schema=schema,physical_VI_integration_rows=rows,expected_native_probes=expected,
        fixed_physical_VI_contract=scene.metadata.get('fixed_VI_observation_contract'),
        legacy_native_VI_contract=scene.metadata.get('legacy_native_VI_observation_contract'),
        VI_observation_roles=scene.metadata['VI_observation_roles'],
        copper_compiled_support=copper,
        material_native_components={kind:{name:dict(min=float(value.min()),max=float(value.max()),sha256=tensor_signature(value))
            for name,value in edge[kind].items()} for kind in ('eps','mu','sigma_e','sigma_m')},
        observation_and_storage=dict(field_real_bytes=field_bytes,native_plane_complex64_bytes=plane_bytes,
            copper_modal_float64_bytes=state_bytes,copper_side_complex128_bytes=spectral_bytes,
            known_tensor_subtotal_bytes=field_bytes+plane_bytes+state_bytes+spectral_bytes,
            minimum_free_gpu_bytes=12*2**30,minimum_available_host_RAM_bytes=16*2**30,minimum_free_disk_bytes=10*2**30,
            scope='Conservative operational reservations; root checks actual shared-job headroom. Not a measured peak or timing guarantee.'),
        source_before=identity,source_after=execution_identity(),actual_field_advance=False,actual_CUDA_preparation=False,
        CUDA_initialized=torch.cuda.is_initialized(),CUDA_initialization_unchanged=torch.cuda.is_initialized()==initial_cuda,physical_release_qualified=False,
        scope='One fixed physical case and two realbasis drives. Pure-control and mesh functional preparation only; no GPU/attenuation/continuum acceptance.')
    assert record['source_before']==record['source_after'] and record['CUDA_initialization_unchanged']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(encode(record),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(metal=case.metal,sigma=case.sigma,dx=case.dx,grid=record['grid_shape'],dt=dt,N=count,split=split,
        known_tensor_bytes=record['observation_and_storage']['known_tensor_subtotal_bytes'],Cu_faces=len(copper))),flush=True)
if __name__=='__main__':main()
