"""Physical signed SI state/basis checks, not a modal/attenuation extractor."""
from pathlib import Path
import json
import numpy as np
from scripts.dev.fdtd.printed_line_two_length import real_basis,physical_transfer,decode
from scripts.dev.fdtd.printed_line_broadband import encode

FREQUENCIES=(.8e9,.9e9,1e9,1.1e9,1.2e9)

def channels(kind,raw):
    raw=np.asarray(raw,dtype=np.complex128)
    count=3 if kind=='cpw' else 6 if kind=='differential_pair' else None
    if count is None or raw.shape!=(count,2,5) or not np.isfinite(raw).all():
        raise ValueError('A complete finite original station/row/frequency SI state is required.')
    if kind=='cpw':return raw[:,0,None,:],raw[:,1,None,:]
    x=raw.reshape(3,2,2,5)
    return np.stack((x[:,0,0]-.5*x[:,1,0],x[:,1,0]),axis=1),np.stack((x[:,0,1],x[:,1,1]-.5*x[:,0,1]),axis=1)


def family_report(kind,records,states):
    n=1 if kind=='cpw' else 2 if kind=='differential_pair' else None
    if n is None or len(records)!=2*n or len(states)!=2*n:
        raise ValueError('Two scalar or four coupled genuinely independent drive columns are required.')
    basis=real_basis(2*n);first=records[0]
    source_columns=[];voltage=[];current=[]
    for index,(record,state) in enumerate(zip(records,states)):
        if record['kind']!=kind or record['drive_index']!=index or tuple(record['frequencies'])!=FREQUENCIES or record['planned_steps']!=1024:
            raise ValueError('Complete ordered drive/frequency/horizon axes cannot be truncated or reordered.')
        if not np.array_equal(np.asarray(record['amplitudes']),basis[index]):
            raise ValueError('The declared real nonzero source basis cannot change.')
        for key in ('case','groups','dt_s','source_identity','operator_without_drive'):
            if record[key]!=first[key]:raise ValueError('A physical/source/operator input differs across drives: '+key)
        groups=record['groups'];ports=record['source_spectra']
        if len(groups)!=2*n or len({p for group in groups for p in group})!=4 or set(ports)!={p for group in groups for p in group}:
            raise ValueError('Every original balanced source group and physical port must be present exactly once.')
        grouped=[]
        for group in groups:
            spectra=[]
            for port in group:
                entry=ports[port]
                values=np.asarray(entry['thevenin_peak_voltage_V'],dtype=np.complex128)
                available=np.asarray(entry['available_power_W'],dtype=float)
                if values.shape!=(5,) or available.shape!=(5,) or not np.isfinite(values).all() or not np.isfinite(available).all() or np.any(available<=0) or entry['source_resistance_ohm']!=50.:
                    raise ValueError('Actual complex source spectra/available power must retain physical units and the complete axis.')
                spectra.append(values)
            if any(not np.array_equal(spectra[0],values) for values in spectra[1:]):
                raise ValueError('Balanced physical sources must retain the same signed complex drive, not just its magnitude.')
            grouped.append(spectra[0])
        source_columns.append(grouped)
        V,I=channels(kind,state);voltage.append(V);current.append(I)
    source=np.asarray(source_columns).transpose(2,1,0)
    V=np.stack(voltage,axis=-1);I=np.stack(current,axis=-1)
    positions=np.asarray(first['case']['positions'],dtype=float)
    if positions.shape!=(3,) or not np.all(np.diff(positions)>0):raise ValueError('Three fixed ordered physical stations are required.')
    frequencies=[]
    for f,frequency in enumerate(FREQUENCIES):
        source_rank=int(np.linalg.matrix_rank(source[f]));source_condition=float(np.linalg.cond(source[f]))
        if source_rank!=2*n or not np.isfinite(source_condition) or source_condition>1e8:
            raise ValueError('Actual source spectra do not resolve the complete complex source basis.')
        states_scaled=np.concatenate((V[:,:,f],50.*I[:,:,f]),axis=1)
        condition=[float(np.linalg.cond(X)) for X in states_scaled]
        rank=[int(np.linalg.matrix_rank(X)) for X in states_scaled]
        if any(r!=2*n for r in rank) or not all(np.isfinite(c) and c<=1e8 for c in condition):
            raise ValueError('Signed physical VI states do not resolve a full rank/conditioned complete basis.')
        transfer=[]
        for low,high in ((0,1),(1,2),(0,2)):
            value=physical_transfer(V[[low,high],:,f,:],I[[low,high],:,f,:])
            transfer.append(dict(positions_m=positions[[low,high]],**value))
        frequencies.append(dict(frequency_Hz=frequency,actual_complex_grouped_source_voltage_V=source[f],actual_source_rank=source_rank,actual_source_condition=source_condition,physical_state_rank=rank,physical_state_condition_50ohm=condition,physical_transfer=transfer))
    return dict(schema='complete-real-basis-native-negative-VI-prefix/1',kind=kind,drive_columns=2*n,frequencies=FREQUENCIES,records=frequencies,
        physical_voltage_V=V,physical_current_A=I,signed_peak_VI_power_W=.5*np.real(np.sum(V*np.conj(I),axis=1)),
        complete_source_and_physical_state_basis=True,full64_tail_qualified=False,mode_qualified=False,S_parameter_qualified=False,attenuation_qualified=False,
        scope='Independent source columns resolve a fixed single-DUT signed SI state operator. A1024step finite-window prefix is not settled modal scattering, full64ns or two independent physical lengths. ScalingI by50 only conditions units; it supplies no modal impedance or de-embedding phase.')


def review_family(folder,kind):
    folder=Path(folder);count=2 if kind=='cpw' else 4
    records=[];states=[]
    for index in range(count):
        p=folder/f'{kind}-drive{index}'
        records.append(decode(json.loads((p/'record.json').read_text())))
        import torch
        states.append(torch.load(p/'actual-VI.pt',map_location='cpu',weights_only=False)['state'].numpy())
    value=family_report(kind,records,states)
    path=folder/f'{kind}-complete-prefix.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(encode(value),indent=2,allow_nan=False)+'\n')
    return value
