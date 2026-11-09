"""Independent engine single-edge Ampere/source-clock proof on a CPU grid."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scripts.dev.fdtd.matched_current_reference import compile_known_current_edges,install_known_current_sources,known_current_spectrum


def run(output,*,eps_r=1.,sigma=0.):
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    dl=.001;dt=.2*dl/299792458.;frequency=30e9;steps=96;amplitude=1e-6
    epsilon=8.8541878128e-12*eps_r
    axes=(np.arange(7)*dl,)*3
    edges=compile_known_current_edges(axes,axis=2,path=(2*dl,3*dl),
        aperture_bounds={0:(1.5*dl,2.5*dl),1:(1.5*dl,2.5*dl)},dt=dt,epsilon=epsilon,conductivity=sigma)
    assert len(edges)==1
    engine=openEMS(NrTS=steps,EndCriteria=0.,TimeStep=dt)
    engine.SetCustomExcite(f'sin(2*pi*{frequency:.17g}*t)',frequency,100e9)
    engine.SetBoundaryCond(['PEC']*6)
    csx=ContinuousStructure();engine.SetCSX(csx)
    for axis,values in enumerate(axes):csx.GetGrid().AddLine(axis,values)
    if eps_r!=1. or sigma:
        material=csx.AddMaterial('uniform',epsilon=eps_r,kappa=sigma)
        material.AddBox([0.,0.,0.],[6*dl]*3,priority=0)
    install_known_current_sources(csx,edges,amplitude=amplitude)
    voltage=csx.AddProbe('edge_voltage',p_type=0,weight=1.,over_sampling=1000000)
    voltage.AddBox([2*dl,2*dl,2*dl],[2*dl,2*dl,3*dl])
    current=csx.AddProbe('edge_H_circulation',p_type=1,weight=1.,norm_dir=2,over_sampling=1000000)
    current.AddBox([1.5*dl,1.5*dl,2.5*dl],[2.5*dl,2.5*dl,2.5*dl])
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    engine.Run(str(output),cleanup=False,verbose=0,numThreads=1)
    e=np.loadtxt(output/'edge_voltage',comments='%');h=np.loadtxt(output/'edge_H_circulation',comments='%')
    source=np.loadtxt(output/'et')
    count=min(len(e)-1,len(h),len(source))
    n=np.arange(count)
    np.testing.assert_allclose(e[:count,0],n*dt,rtol=1e-10,atol=1e-22)
    np.testing.assert_allclose(h[:count,0],(n+.5)*dt,rtol=1e-10,atol=1e-22)
    capacity=epsilon*dl;conductance=sigma*dl
    inferred=capacity*np.diff(e[:count+1,1])/dt+conductance*(e[1:count+1,1]+e[:count,1])/2-h[:count,1]
    expected=amplitude*source[:count,1]
    relative=float(np.linalg.norm(inferred-expected)/np.linalg.norm(expected))
    exact_clock=(n+.5)*dt
    direct=known_current_spectrum(source[:count,0],source[:count,1],[frequency],dt=dt,amplitude=amplitude)
    reconstructed=2*dt*(np.exp(2j*np.pi*frequency*exact_clock)@inferred)
    phase_error=float(abs(np.angle(reconstructed/direct[0])))
    without_half_clock=2*dt*amplitude*(np.exp(2j*np.pi*frequency*n*dt)@source[:count,1])
    omitted_phase=float(abs(np.angle(direct[0]/without_half_clock)))
    report=dict(schema='independent-known-current-source-clock/1',eps_r=eps_r,sigma_s_m=sigma,
        dt_s=dt,frequency_hz=frequency,samples=count,source_edges=len(edges),known_current_peak_a=amplitude,
        ampere_time_domain_relative_error=relative,phasor_phase_error_rad=phase_error,
        omitted_half_clock_phase_rad=omitted_phase,expected_half_clock_phase_rad=math.pi*frequency*dt,
        electric_probe_clock='E[n] at n*dt',magnetic_probe_clock='H[n] at (n+.5)*dt',
        source_clock='et f[n] is impressed field-work current at (n+.5)*dt and advances E[n+1]',
        acceptance=dict(ampere_time_domain_relative_error=3e-5,phasor_phase_error_rad=3e-5),
        passed=relative<3e-5 and phase_error<3e-5)
    (output/'control.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    assert report['passed'],report
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--eps-r',type=float,default=1.);parser.add_argument('--sigma',type=float,default=0.)
    args=parser.parse_args();run(args.output,eps_r=args.eps_r,sigma=args.sigma)
