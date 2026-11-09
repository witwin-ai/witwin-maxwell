"""Private fixed C2 startup control for the retained annular TE1 diagnostic.

Both native CW quadratures receive the same declared envelope at their actual
electric/magnetic sample times. This does not register a public mapped source.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scripts.dev.fdtd import bend_annulus_time_control as capture
from scripts.dev.fdtd.bend_annulus_em_audit import decoded
from scripts.dev.fdtd.copper_cylinder_reference import MU_0,shell_scattering


def envelope(time_value,ramp_seconds):
    q=max(0.,min(1.,float(time_value)/ramp_seconds))
    return q*q*q*(10.+q*(-15.+6.*q))


def run(output,ramp_ns):
    if ramp_ns not in (20.,40.):
        raise ValueError('This predeclared control permits only 20/40 ns ramps.')
    from witwin.maxwell.fdtd.excitation import temporal
    paths=(Path(__file__),Path(capture.__file__),Path(shell_scattering.__code__.co_filename))
    def binding():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in paths}
    before=binding();original_signal=temporal._resolve_cw_signal;original_time=capture.mw.TimeConfig
    def signal(omega,source_time,time_value):
        factor=envelope(time_value,ramp_ns*1e-9)
        return tuple(factor*value for value in original_signal(omega,source_time,time_value))
    def timing(**kwargs):
        return original_time(**dict(kwargs,transient_cycles=ramp_ns+24.,steady_cycles=20.))
    temporal._resolve_cw_signal=signal;capture.mw.TimeConfig=timing
    try:
        result=capture.run(output)
    finally:
        temporal._resolve_cw_signal=original_signal;capture.mw.TimeConfig=original_time
    after=binding()
    if before!=after:raise RuntimeError('Smooth startup diagnostic changed during execution.')
    frequency=result['frequency_hz'];sigma=5.8e7;delta=math.sqrt(2/(2*math.pi*frequency*MU_0*sigma))
    scattering=shell_scattering(result['radius_m'],delta,frequency,sigma,order=1,polarization='te')
    incoming=decoded(result['incoming']);outgoing=decoded(result['outgoing'])
    reference_absorption=float(np.vdot(incoming,(np.eye(2)-scattering.conj().T@scattering)@incoming).real/np.vdot(incoming,incoming).real)
    measured_absorption=float(1-np.vdot(outgoing,outgoing).real/np.vdot(incoming,incoming).real)
    result.update(schema='private-annulus-fixed-smooth-start/1',ramp_ns=ramp_ns,post_ramp_settling_ns=24.,observation_ns=20.,
        envelope='C2 quintic 6q^5-15q^4+10q^3; q=clip(t/ramp,0,1); both CW quadratures',
        smooth_controller_binding_before=before,smooth_controller_binding_after=after,
        coherent_reference_absorption=reference_absorption,measured_flux_absorption=measured_absorption,
        measured_flux_relative_error=abs(measured_absorption-reference_absorption)/reference_absorption,
        direct_heat_relative_error=abs(result['direct_heat_absorption']-reference_absorption)/reference_absorption,
        qualification='One directional counterexample only; no complete two-port S-matrix qualification.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ramp-ns',type=float,choices=(20.,40.),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.ramp_ns)
