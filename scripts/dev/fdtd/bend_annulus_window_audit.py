"""Fixed-window and low-frequency FFT audit of retained native annular fields.

All Fourier sums run in GPU complex128 on unchanged native time samples. The
windows are declared here before execution; none replaces the physical S gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch


WINDOWS_NS=tuple((start,start+20)for start in (0,4,8,12,16,20,24))+tuple((44-length,44)for length in (8,12,16))


def phasor(record,start_ns,end_ns,dt,frequency):
    buffer=record['values'];n=record['steps'].to(dtype=torch.float64,device=buffer.device)
    lo=round(start_ns*1e-9/dt);hi=round(end_ns*1e-9/dt);valid=(n>=lo)&(n<hi)
    if int(valid.sum())!=hi-lo:raise ValueError('The declared window lacks retained native samples.')
    w=.5*(1-torch.cos(2*torch.pi*(n[valid]-lo)/(hi-lo)))
    time=(n[valid]-(.5 if record['component'].startswith('H') else 0))*dt
    return 2*torch.sum(buffer[valid].to(torch.complex128)*w*torch.exp(2j*torch.pi*frequency*time))/w.sum()


def audit(report_path,output):
    report=json.loads(report_path.read_text());raw_path=Path(report['raw_samples'])
    paths=(Path(__file__),report_path,raw_path)
    def binding():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in paths}
    before=binding()
    if before[str(raw_path)]!=report['raw_samples_sha256']:raise ValueError('Retained time samples do not match the original run.')
    if report['source_binding_before']!=report['source_binding_after']:raise ValueError('Native source changed during the original run.')
    raw=torch.load(raw_path,map_location='cuda',weights_only=False)
    dt=report['dt_s'];frequency=report['frequency_hz'];rows=[]
    for start,end in WINDOWS_NS:
        values={key:phasor(record,start,end,dt,frequency)for key,record in raw.items()};flux={}
        for side in ('left','right'):
            e=values[side+'_e_0'];h=(values[side+'_h_m']+values[side+'_h_p'])/2
            flux[side]=float(-.5*(e*h.conj()).real)*(4*report['spacing_m'])**2*math.cos(math.pi*frequency*dt)
        rows.append(dict(start_ns=start,end_ns=end,flux_w=flux,net_flux_w=flux['left']-flux['right'],
            phasors={key:[float(value.real),float(value.imag)]for key,value in values.items()}))
    peaks={}
    for key in ('left_h_m','left_e_0','right_h_m'):
        record=raw[key];n=record['steps'].to(device='cuda',dtype=torch.float64)
        values=record['values'][n*dt>=20e-9].to(torch.complex128)
        weights=torch.hann_window(values.numel(),periodic=True,device='cuda',dtype=torch.float64)
        spectrum=torch.fft.fft(values*weights)
        frequencies=torch.fft.fftfreq(values.numel(),dt,device='cuda',dtype=torch.float64)
        band=(frequencies<-.1e9)&(frequencies>-.8e9);indices=torch.where(band)[0]
        selected=indices[torch.argsort(abs(spectrum[indices]),descending=True)[:8]]
        carrier=abs(phasor(record,20,44,dt,frequency))
        peaks[key]=dict(cw_amplitude=float(carrier),peaks=[dict(frequency_hz=float(-frequencies[index]),
            amplitude_relative_to_1ghz=float(2*abs(spectrum[index])/weights.sum()/carrier))for index in selected])
    after=binding()
    if before!=after:raise RuntimeError('Retained samples or window audit changed during analysis.')
    result=dict(schema='private-annulus-fixed-window-audit/1',binding_before=before,binding_after=after,
        windows=rows,low_frequency_fft=peaks,device=torch.cuda.get_device_name(),
        note='Fixed declared windows; no transient subtraction, source amplitude fit, or physical gate replacement.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();audit(args.report,args.output)
