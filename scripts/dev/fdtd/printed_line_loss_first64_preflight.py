"""CPU geometry/time-bound/resource forecast; no FDTD engine or field advance."""
import dataclasses, hashlib, json, math, os, shutil, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase, DRIVE_BASIS, build_case, make_simulation, execution_identity, tensor_signature
from scripts.dev.fdtd.printed_line_broadband import encode
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.copper_foil import compile_copper_faces, compile_copper_response
from witwin.maxwell.compiler.ports import compile_ports
from witwin.maxwell.constants import C_0, EPSILON_0, MU_0
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.fdtd.runtime.timestep import compile_fdtd_time_step

output=Path(os.environ['PCB_FIRST64_PREFLIGHT_OUTPUT'])
if output.exists():raise FileExistsError(output)
start=execution_identity();case=PilotCase();scene,groups,bounds,schema,rows,expected,tails=build_case(case)
prepared=prepare_scene(scene);model=prepared.compile_materials();edge=model['edge_components']
simulation=make_simulation(scene,groups,DRIVE_BASIS[0],case,graph=True,steps=1,device='cpu')
frequency=simulation._fdtd_time_step_frequency(prepared)
minimum=[float(getattr(prepared,f'd{axis}_primal64').min()) for axis in 'xyz']
# This is a preparation-only scalar coefficient context, not an FDTD solver.
# No E/H fields, native kernels, actual counters, or stand-in propagation exist.
context=SimpleNamespace(scene=prepared,c=C_0,eps0=EPSILON_0,mu0=MU_0,full_aniso_enabled=False,
    dt=FDTD.auto_dt(None,*minimum,frequency),_requested_source_frequency=frequency)
for component in ('Ex','Ey','Ez'):setattr(context,'eps_'+component,edge['eps'][component]*EPSILON_0)
for component in ('Hx','Hy','Hz'):setattr(context,'mu_'+component,edge['mu'][component]*MU_0)
compile_fdtd_time_step(context,model)
dt=float(context.dt);steps=math.ceil(case.duration/dt);checkpoint=math.floor(case.checkpoint_time/dt)
shapes={component:tuple(edge['eps' if component.startswith('E') else 'mu'][component].shape) for component in ('Ex','Ey','Ez','Hx','Hy','Hz')}
faces=compile_copper_faces(prepared,device='cpu');copper=[]
for face in faces:
 response=compile_copper_response(face.material)
 copper.append(dict(name=face.name,component='E'+'xyz'[face.component],normal_axis='xyz'[face.axis],edge_count=len(face.indices),rates=response.rates.numel(),B_shape=list(response.state_space.B.shape),D_shape=list(response.state_space.D.shape),constitutive_relative_error_bound=response.relative_error_bound))
ports=compile_ports(prepared,device='cpu')
source_contract=[]
for amplitudes in DRIVE_BASIS:
 config=make_simulation(scene,groups,amplitudes,case,graph=True,steps=steps,device='cpu')
 source_contract.append([dict(port=excitation.port_name,amplitude=float(excitation.amplitude),source_impedance=float(excitation.source_impedance),source_time=dataclasses.asdict(excitation.source_time)) for excitation in config.excitations])
inverse=np.linalg.inv(np.array(DRIVE_BASIS).T)
np.testing.assert_allclose(np.array(DRIVE_BASIS).T@inverse,np.eye(2),rtol=0,atol=4e-16)
assert np.isrealobj(inverse)
field_bytes=sum(math.prod(shape)*4 for shape in shapes.values())
plane_samples=sum(len(selection['planes'])*math.prod(selection['full_shape'][:2]) for selection in schema.values())
plane_complex_bytes=plane_samples*len(case.frequencies)*8
foil_spectrum_bytes=sum(2*entry['edge_count']*len(case.frequencies)*16 for entry in copper)
foil_state_bytes=sum(entry['edge_count']*entry['rates']*8 for entry in copper)
# Conservative capacity reservations, not an assertion of measured peak memory.
resources=dict(field_real_bank_bytes=field_bytes,native_plane_complex64_spectra_bytes=plane_complex_bytes,
 copper_complex128_side_spectra_bytes=foil_spectrum_bytes,copper_float64_modal_state_bytes=foil_state_bytes,
 minimum_free_gpu_bytes=8*2**30,minimum_available_host_ram_bytes=16*2**30,minimum_free_disk_bytes=8*2**30,
 analytic_tensor_subtotal_bytes=field_bytes+plane_complex_bytes+foil_spectrum_bytes+foil_state_bytes,
 bound_scope='8GiB VRAM/16GiB RAM/8GiB disk are conservative operational reservations, not allocator peak guarantees; root checks actual headroom and monitors real job. No timing claim.',
 two_drive_result_checkpoint_sidecar_disk_capacity_reserved_bytes=8*2**30,
 full_field_DFT=False,full_vector_norm_new_pilot_storage=False,plots_written=False)
record=dict(schema='first64ns-combined-stripline-CPU-preflight/1',created_unix_s=time.time(),case=dataclasses.asdict(case),
 groups=groups,source_basis=DRIVE_BASIS,pure_left_right_real_inverse_coefficients=inverse.tolist(),
 source_inverse_definition='columns c solve declared_source_basis.T @ c = pure_left_or_right; actual independently accumulated source phasors must confirm cancellation. No constant-i Gaussian or response-fit assumption.',
 source_contract=source_contract,raw_calendar_port_contract=[dict(name=port.name,positive=port.positive,negative=port.negative,voltage_aperture=dict(position=port.voltage_aperture.position,size=port.voltage_aperture.size),current_surface=dict(position=port.current_surface.position,size=port.current_surface.size),termination=port.termination) for port in scene.ports],bounds_m=bounds,actual_prepared_grid_node_shape=[prepared.Nx,prepared.Ny,prepared.Nz],native_component_shapes=shapes,
 grid_node_sha256={axis:tensor_signature(getattr(prepared,axis+'_nodes64')) for axis in 'xyz'},
 CPU_compiled_time_step_forecast_s=dt,CPU_time_step_metadata=context._time_step_metadata,
 planned_steps_forecast=steps,checkpoint_step_forecast=checkpoint,planned_duration_forecast_s=steps*dt,checkpoint_time_forecast_s=checkpoint*dt,
 time_semantics='Fresh actual CUDA Scene->Simulation.prepare supplies authoritative dt; run32ns checkpoint with planned full64nsN, freshsameNprepare and diskresume. Forecast must agree with actual runtime before sourceadvance; no extension of an old Result.',
 native_plane_schema=schema,native_probe_count=len(expected),native_probe_contract=expected,physical_VI_integration_rows=rows,
 copper_compiled_support=copper,port_compiler_signatures={port.port_name:tensor_signature(dataclasses.asdict(port)) for port in ports},
 observation_and_storage=resources,resources_available=dict(disk_free_bytes=shutil.disk_usage(output.parent).free,mem_available_bytes=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024),
 source_before=start,source_after=execution_identity(),actual_CUDA_runtime_preparation_executed=False,field_advance_executed=False,cuda_initialized=torch.cuda.is_initialized(),
 physical_release_qualified=False,scope='One combined Cu35um/sigma.01 material and one .12m source separation, two realbasis drives only; no expansion/fitting/two-length attenuation claim.')
assert record['source_before']==record['source_after'];assert not record['cuda_initialized']
output.write_text(json.dumps(encode(record),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(shape=record['actual_prepared_grid_node_shape'],dt_forecast=dt,steps=steps,checkpoint=checkpoint,analytic_tensor_subtotal_bytes=resources['analytic_tensor_subtotal_bytes'],copper=copper)))
