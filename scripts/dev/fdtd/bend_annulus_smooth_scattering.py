"""Run the unchanged annular Hz-only S gates with predeclared C2 startup.

Use only a frozen native source archive. This four-run controller preserves the
original Hankel projection, reference, geometry and physical acceptance gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from scripts.dev.fdtd import bend_cylinder_native_control as control
from scripts.dev.fdtd.bend_annulus_smooth_start import envelope


def campaign(output,*,spacing,ramp_ns):
    from witwin.maxwell.fdtd.excitation import temporal
    if ramp_ns not in (20.,40.):raise ValueError('Use a predeclared 20/40 ns startup.')
    paths=(Path(__file__),Path(envelope.__code__.co_filename))
    def binding():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in paths}
    before=binding();original_signal=temporal._resolve_cw_signal;original_time=control.mw.TimeConfig
    def signal(omega,source_time,time_value):
        factor=envelope(time_value,ramp_ns*1e-9)
        return tuple(factor*value for value in original_signal(omega,source_time,time_value))
    def timing(**kwargs):
        return original_time(**dict(kwargs,transient_cycles=ramp_ns+24.,steady_cycles=20.))
    temporal._resolve_cw_signal=signal;control.mw.TimeConfig=timing
    try:
        result=control.campaign(output,spacing=spacing,orders=(1,),polarizations=('te',),
            radius=.3,frequency=1e9,public_metric=True,save_results=True)
    finally:
        temporal._resolve_cw_signal=original_signal;control.mw.TimeConfig=original_time
    after=binding()
    if before!=after:raise RuntimeError('The fixed-startup scattering controller changed during execution.')
    result.update(startup_controller_binding_before=before,startup_controller_binding_after=after,
        ramp_ns=ramp_ns,post_ramp_settling_ns=24.,observation_ns=20.,
        envelope='C2 quintic, both native CW quadratures at their actual sample times',
        acceptance_contract='Original Hz-only Hankel S-matrix gates unchanged; no selected window or vacuum correction.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spacing',type=float,required=True)
    parser.add_argument('--ramp-ns',type=float,choices=(20.,40.),required=True)
    args=parser.parse_args();result=campaign(args.output,spacing=args.spacing,ramp_ns=args.ramp_ns)
    if not all(row['passed']for row in result['rows']):
        raise SystemExit('Native annulus physical acceptance failed; original gates and raw evidence retained.')
