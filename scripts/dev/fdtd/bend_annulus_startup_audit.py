"""Compare the same late-time cavity band in unchanged native startup runs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.dev.fdtd.bend_annulus_em_audit import project_em,decoded


def audit(reports,output):
    paths=[Path(__file__),Path(project_em.__code__.co_filename)];metadata=[]
    for path in reports:
        report=json.loads(path.read_text());raw=Path(report['raw_samples'])
        if report['source_binding_before']!=report['source_binding_after']:
            raise ValueError('The native source changed during a retained startup run.')
        if report.get('smooth_controller_binding_before')!=report.get('smooth_controller_binding_after'):
            raise ValueError('The private startup controller changed during execution.')
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=report['raw_samples_sha256']:
            raise ValueError('Retained native time samples changed after execution.')
        paths.extend((path,raw));metadata.append((path,report))
    def binding():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in paths}
    before=binding();rows=[]
    for path,report in metadata:
        raw=torch.load(report['raw_samples'],map_location='cuda',weights_only=False)
        record=raw['left_h_m'];dt=report['dt_s'];end=report['steps']*dt
        n=record['steps'].to(device='cuda',dtype=torch.float64)
        values=record['values'][n*dt>=end-24e-9].to(torch.complex128)
        weights=torch.hann_window(values.numel(),device='cuda',dtype=torch.float64,periodic=True)
        spectrum=torch.fft.fft(values*weights)
        frequencies=-torch.fft.fftfreq(values.numel(),dt,device='cuda',dtype=torch.float64)
        indices=torch.where((frequencies>.1e9)&(frequencies<.8e9))[0]
        peak=indices[torch.argmax(abs(spectrum[indices]))]
        carrier=report['comparison']['left_h_m']['complex128']
        amplitude=abs(complex(carrier['real'],carrier['imag']))
        projection={}
        for label in ('complex128','native_point'):
            amplitudes=[]
            for side,u in (('left',-report['radius_m']/6),('right',report['radius_m']/6)):
                def value(key):return decoded(report['comparison'][side+'_'+key][label]).item()
                amplitudes.append(project_em(value('e_0'),(value('h_m')+value('h_p'))/2,
                    report['radius_m']+u,report['radius_m'],report['frequency_hz'],1))
            incoming=np.asarray((amplitudes[0][0],amplitudes[1][1]))
            outgoing=np.asarray((amplitudes[0][1],amplitudes[1][0]))
            absorption=float(1-np.vdot(outgoing,outgoing).real/np.vdot(incoming,incoming).real)
            projection[label]=dict(absorption=absorption)
            if 'coherent_reference_absorption' in report:
                reference=report['coherent_reference_absorption']
                projection[label]['relative_error']=abs(absorption-reference)/reference
        rows.append(dict(report=str(path),ramp_ns=report.get('ramp_ns',0.),start_ns=(end-24e-9)*1e9,
            end_ns=end*1e9,peak_hz=float(frequencies[peak]),
            peak_amplitude_over_carrier=float(2*abs(spectrum[peak])/weights.sum()/amplitude),
            em_projection=projection))
    after=binding()
    if before!=after:raise RuntimeError('Startup analysis or retained inputs changed during execution.')
    result=dict(schema='private-annulus-late-startup-fft/1',binding_before=before,binding_after=after,
        selection='Last complete 24 ns, predeclared 0.1-0.8 GHz band; native complex64 fields, GPU complex128 FFT',
        qualification='Diagnostic only; no subtraction, fit, gate replacement, or absolute CW source calibration.',rows=rows)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.reports,args.output)
