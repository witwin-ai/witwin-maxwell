"""Reintegrate retained native phasors in complex128 without rerunning fields."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from benchmark.scenes.rf.printed_lines import printed_line_scene
from scripts.dev.fdtd.printed_line_broadband import configuration,load_helper,encode
from scripts.dev.fdtd.printed_line_physical_modes import coupled_physical_state,coupled_transfer


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous retained-field reductions.')
    before=digest(args.input);record=json.loads(args.input.read_text());options=SimpleNamespace(**record['configuration'])
    driver=digest(__file__);fixture=Path(inspect.getfile(printed_line_scene))
    helper,helper_path=load_helper(options.kind);state_helper=Path(inspect.getfile(coupled_transfer))
    if digest(fixture)!=record['fixture_sha256'] or digest(helper_path)!=record['native_contour_helper_sha256'] or digest(state_helper)!=record['state_helper_sha256']:
        raise ValueError('Retained native samples require their exact original physical fixture and integration helpers.')
    if 'integration_rows' in record:rows=record['integration_rows']
    else:
        scene=configuration(options,'cpu')
        if options.kind=='differential_pair':_,rows,expected=helper.probes(scene,options.dx,options.longitudinal_dl,options.padding,options.positions,options.metal)
        else:
            _,single,expected=helper.native_probe_monitors(scene,options.kind,options.dx,options.longitudinal_dl,options.padding,options.positions,options.metal)
            rows=[[r] for r in single]
        for name,(field,index) in expected.items():
            source=record['native_probe_spectra'][name]
            if source['component']!=field or list(index)!=source['index']:raise ValueError('Physical measurement geometry changed.')
    spectra={name:np.array([[complex(v['real'],v['imag']) for v in drive] for drive in source['all_drive_frequency_phasors']],dtype=np.complex128)
        for name,source in record['native_probe_spectra'].items()}
    def integral(path):return sum(spectra[name]*weight for name,_,weight in path)
    states=[]
    for station in rows:
        pairs=[(integral(v),integral(i)) for v,i in station]
        states.append(coupled_physical_state(pairs[0][0],pairs[1][0],pairs[0][1],pairs[1][1]) if options.kind=='differential_pair' else np.stack(pairs[0]))
    states=np.asarray(states) # station,state,drive,frequency
    frequencies=[]
    for index,frequency in enumerate(record['frequencies_hz']):
        state=states[:,:,:,index]
        transfers=coupled_transfer(state[:,:2],state[:,2:],options.positions) if options.kind=='differential_pair' else helper.state_transfer(state[:,0],state[:,1],options.positions)
        frequencies.append({'frequency_hz':frequency,'actual_physical_state':state,'state_transfer':transfers})
    output={'schema':'retained-native-line-complex128-reduction/1','input_sha256':before,'input_unchanged':before==digest(args.input),
        'driver_sha256':driver,'driver_unchanged':driver==digest(__file__),'field_reexecution':False,
        'field_source_sha256':record['source']['source_sha256'],'fixture_sha256':record['fixture_sha256'],
        'native_contour_helper_sha256':record['native_contour_helper_sha256'],'state_helper_sha256':record['state_helper_sha256'],
        'integration_rows':rows,'frequency_records':frequencies,'scope':'Only SI quadrature and transfer algebra use complex128; all stored native point phasors, source, geometry, fields and time records are unchanged.'}
    args.output.write_text(json.dumps(encode(output),indent=2,allow_nan=False)+'\n')
    if not output['input_unchanged'] or not output['driver_unchanged']:raise RuntimeError('Retained-field reduction identity changed.')


if __name__=='__main__':main()
