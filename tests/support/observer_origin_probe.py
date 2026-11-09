"""CPU-only public-API measurements against a closed finite Yee recurrence."""
import os
import sys, json, math, cmath
from pathlib import Path
import numpy as np
import torch
import witwin.maxwell as mw
torch.set_num_threads(1)

def closed_record(A,b,omega,dt,count,alpha):
    r=np.exp(-1j*omega*dt)
    inv=np.linalg.solve(r*np.eye(2)-A,b)
    eigenvalues,eigenvectors=np.linalg.eig(A.astype(complex))
    inverse=np.linalg.inv(eigenvectors)
    return np.array([((r**(n+1)*np.eye(2)-eigenvectors@np.diag(eigenvalues**(n+1))@inverse)@inv*np.exp(-1j*alpha*omega*dt)).real for n in range(count)])

def run(kind, count=64, normalized=False):
    dx=.05; L=.4; f=2e9; dt=9.5238e-12; N=count
    z=np.arange(9,dtype=float)*dx
    if kind=='J':
        coords=z; profile=np.sin(math.pi*coords/L); component='Jx'
    else:
        coords=(np.arange(10,dtype=float)-.5)*dx
        profile=np.cos(math.pi*coords/L); component='My'
    # Nodal y endpoint control cells are clipped to half coverage by this compiler.
    # Compensate in the declared density so the COMPILED current is uniform in y.
    values=np.broadcast_to(profile[None,None,:],(2,3,len(profile))).copy()
    values*=np.array([2.,1.,2.])[None,:,None]
    scene=mw.Scene(domain=mw.Domain(bounds=((0.,.1),(0.,.1),(0.,L))),
        grid=mw.GridSpec.uniform(dx),device=os.environ.get('OBSERVER_ORIGIN_DEVICE', 'cpu'),
        boundary=mw.BoundarySpec.pec().with_faces(y_low='pmc',y_high='pmc'))
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(([-.05,.15],[0.,.05,.1],coords),{component:values}),source_time=mw.CW(frequency=f)))
    scene.add_monitor(mw.PointMonitor(name='probe',position=(.025,.05,.1),fields=('Ex','Hy')))
    sim=mw.Simulation.fdtd(scene,frequencies=(f,),backend='cpu_reference' if torch.device(scene.device).type == 'cpu' else 'native',cuda_graph=False,
        field_dtype=torch.float32,run_time=mw.TimeConfig(time_steps=N,dt=dt,transient_cycles=0),
        full_field_dft=False,spectral_sampler=mw.SpectralSampler(window='none',normalize_source=normalized,observer_dtype=torch.float32 if normalized else torch.float64))
    prepared=sim.prepare()
    result=prepared.run()
    solver=prepared.solver
    raw=result.raw_monitor('probe')
    eps=float(solver.eps_Ex[0,1,2]); mu=float(solver.mu_Hy[0,1,1]); c=1/math.sqrt(eps*mu); eta=math.sqrt(mu/eps)
    actual_dt=solver.dt
    k=2*math.sin(math.pi*dx/(2*L))/dx; a=c*k*actual_dt
    A=np.array([[1-a*a,a],[-a,1.]])
    b=np.array([-actual_dt/eps,0.]) if kind=='J' else np.array([-a*actual_dt*eta/mu,-actual_dt*eta/mu])
    current=closed_record(A,b,2*math.pi*f,actual_dt,N,.5 if kind=="J" else 0.)
    proposal=closed_record(A,b,2*math.pi*f,actual_dt,N,.5 if kind=='J' else 0.)
    kernels=np.exp(2j*math.pi*f*np.arange(N)*actual_dt)
    phase_h=np.exp(-1j*math.pi*f*actual_dt)
    alpha=.5 if kind=='J' else 0.
    source_current=2/N*np.sum(np.cos(2*math.pi*f*(np.arange(N)+alpha)*actual_dt)*kernels)
    source_physical=source_current*np.exp(2j*math.pi*f*actual_dt*alpha)
    row=dict(kind=kind,normalized=normalized,dt=actual_dt,frequency=f,N=N,eps=eps,mu=mu,c=c,eta=eta,kappa=k,A=A.tolist(),b=b.tolist(),indices=raw.get('field_indices'),start=solver._observer_spectral_entries[0]['start_step'],measurements={})
    for comp,index in raw['field_indices'].items():
        zi=index[2]
        if comp=='Ex':
            loc=float(solver.scene.z_nodes64[zi]); shape=math.sin(math.pi*loc/L); channel=0; stagger=1.
        else:
            loc=float(solver.scene.z_half64[zi]); shape=math.cos(math.pi*loc/L)/eta; channel=1; stagger=phase_h
        observed=complex(raw['components'][comp])
        expect=2/N*np.sum(current[:,channel]*kernels)*shape*stagger
        future=2/N*np.sum(proposal[:,channel]*kernels)*shape*stagger
        physical=expect*np.exp(2j*math.pi*f*actual_dt)
        if normalized:
            expect/=source_current;future/=source_current;physical/=source_physical
        row['measurements'][comp]=dict(z=loc,observed=[observed.real,observed.imag],current_closed=[expect.real,expect.imag],proposed_closed=[future.real,future.imag],physical_closed=[physical.real,physical.imag],current_relative_error=abs(observed-expect)/max(abs(expect),1e-30),physical_relative_error=abs(observed-physical)/max(abs(physical),1e-30),proposal_relative_error=abs(observed-future)/max(abs(future),1e-30),current_over_proposed=[(observed/future).real,(observed/future).imag])
    final_e=float(solver.Ex[0,1,2])/math.sin(math.pi*float(solver.scene.z_nodes64[2])/L)
    final_h=float(solver.Hy[0,1,1])*eta/math.cos(math.pi*float(solver.scene.z_half64[1])/L)
    row['final_modal']=[final_e,final_h]
    row['final_closed']=current[-1].tolist()
    if N==1:
        origin={}
        for comp,index in raw['field_indices'].items():
            sample=float(getattr(solver,comp)[tuple(index)])
            observed=complex(raw['components'][comp])
            expected=2*sample*(phase_h if comp=='Hy' else 1)
            if sample == 0:
                continue
            origin[comp]=dict(observed=[observed.real,observed.imag],expected=[expected.real,expected.imag],relative_error=abs(observed-expected)/abs(expected),ratio_to_twice_final=[(observed/(2*sample)).real,(observed/(2*sample)).imag])
        row['origin']=origin
    return row



def port_table_probe():
    from witwin.maxwell.fdtd.ports import prepare_port_spectral_accumulators
    F=2e9
    DT=9.5238e-12
    scene=mw.Scene(domain=mw.Domain(bounds=((0.,.1),(0.,.1),(0.,.4))),grid=mw.GridSpec.uniform(.05),device=os.environ.get('OBSERVER_ORIGIN_DEVICE', 'cpu'),boundary=mw.BoundarySpec.pec().with_faces(y_low='pmc',y_high='pmc'))
    scene.add_port(mw.LumpedPort('feed',negative=(0.,.05,.1),positive=(.1,.05,.1),voltage_path=mw.AxisPath('x'),current_surface=mw.Box(position=(.025,.05,.1),size=(0.,.05,.05)),reference_impedance=50.))
    scene.add_monitor(mw.PointMonitor('probe',position=(.025,.05,.1),fields=('Ex','Hy')))
    sim=mw.Simulation.fdtd(scene,frequencies=(F,),backend='cpu_reference' if torch.device(scene.device).type == 'cpu' else 'native',cuda_graph=False,run_time=mw.TimeConfig(time_steps=4,dt=DT),excitations=mw.PortExcitation('feed',source_time=mw.CW(frequency=F)),full_field_dft=False,spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False,observer_dtype=torch.float64))
    prepared=sim.prepare();s=prepared.solver
    prepare_port_spectral_accumulators(s,4,'none')
    r=s._port_runtimes[0]
    theta=2*math.pi*F*s.dt
    payload=dict(dt=s.dt,frequency=F,theta=theta,voltage_weights=[[z.real,z.imag] for z in r.voltage_phase_weights.reshape(-1).tolist()],current_weights=[[z.real,z.imag] for z in r.current_phase_weights.reshape(-1).tolist()],expected_physical_weights=[[cmath.exp(1j*theta*(n+.5)).real,cmath.exp(1j*theta*(n+.5)).imag] for n in range(4)],first_drive_buffer=float(r.drive_buffer),prepared_grid=[s.Nx,s.Ny,s.Nz])
    return locals()


def port_unit_probe():
    from fractions import Fraction
    ns=port_table_probe()
    s,r=ns['s'],ns['r']
    from witwin.maxwell.fdtd.runtime.stepping import _field_update_block,enforce_pec_boundaries
    from witwin.maxwell.fdtd.ports import apply_port_runtimes,accumulate_port_observers
    theta=2*math.pi*ns['F']*s.dt
    eps=float(s.eps_Ex[0,1,2]);mu=float(s.mu_Hy[0,1,1])
    # Declared two-edge voltage integral and interior dual control cells.
    beta=Fraction(s.dt)*2*Fraction('.05')**2/(Fraction(eps)*Fraction('.05')**3)
    drive=math.cos(theta/2)
    exact_i=Fraction(drive)/(Fraction(50)+beta/2)
    exact_v=beta*exact_i/2
    records=[]
    s._prepare_observers(ns['F'],'none',4)
    for n in range(2):
        _field_update_block(s,n*s.dt)
        apply_port_runtimes(s)
        enforce_pec_boundaries(s)
        accumulate_port_observers(s)
        s.accumulate_observers(n)
        v,i=r.accumulator.phasors()
        fields=s.get_observer_results()['probe']['components']
        records.append(dict(n=n,E=float(s.Ex[0,1,2]),H=float(s.Hy[0,1,1]),V=float(r.lumped.last_voltage_midpoint),I=-float(r.lumped.last_branch_current),reported_V=[float(v.real[0]),float(v.imag[0])],reported_I=[float(i.real[0]),float(i.imag[0])],reported_fields={k:[complex(z).real,complex(z).imag] for k,z in fields.items()},drive=float(r.drive_buffer)))
    one=records[0]
    expected_E=float(Fraction(s.dt)*Fraction('.05')/(Fraction(eps)*Fraction('.05')**3)*exact_i)
    expected_H2=-s.dt/mu*expected_E/.05
    payload=dict(dt=s.dt,frequency=ns['F'],theta=theta,route='public CPU prepare -> unchanged main field/port CPU blocks; excludes CUDA-event wrapper only; no monkeypatch',beta=str(beta),independent_N1_I=float(exact_i),independent_N1_V=float(exact_v),independent_N1_E=expected_E,independent_N2_H=expected_H2,eps=eps,mu=mu,prepared_voltage_weights=r.lumped.voltage_weights.tolist(),records=records,first_V_ratio=[(complex(*one['reported_V'])/(2*one['V'])).real,(complex(*one['reported_V'])/(2*one['V'])).imag],first_I_ratio=[(complex(*one['reported_I'])/(2*one['I'])).real,(complex(*one['reported_I'])/(2*one['I'])).imag])
    return payload
