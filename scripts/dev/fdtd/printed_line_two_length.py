"""Private preparation for independent physical two-length attenuation tests.

No field step runs during import or preparation. Each length is a fresh Scene,
not a pair of stations from one run. Native spectra and SI path weights stay
complex128 during reduction; the electromagnetic solve retains its own dtype.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import numpy as np
from scripts.dev.fdtd.printed_line_broadband import encode
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase, DRIVE_BASIS, run_drive

FREQUENCIES = (.8e9, .9e9, 1e9, 1.1e9, 1.2e9)
SEPARATIONS = (.40, .48)
REFERENCE_OFFSET = .025
KINDS = ('microstrip', 'stripline', 'cpw', 'differential_pair')


def cases():
    return tuple(PilotCase(axial_half_length=.30, separation=length,
        positions=(-length/2+REFERENCE_OFFSET, 0., length/2-REFERENCE_OFFSET),
        probe_reference_dx=.0025, record_legacy_native_vi=True)
        for length in SEPARATIONS)


def real_basis(channels):
    if channels not in (2, 4):
        raise ValueError('The experiment requires two scalar or four coupled physical source groups.')
    return .75*np.eye(channels) + .25*np.ones((channels, channels))


def mixed_transform(voltage, current):
    """Power-dual common/differential change of conductor coordinates, in SI."""
    voltage, current = np.asarray(voltage, dtype=np.complex128), np.asarray(current, dtype=np.complex128)
    if voltage.shape != current.shape or voltage.ndim < 1 or voltage.shape[0] != 2:
        raise ValueError('Both conductor arrays must have first dimension two and identical shapes.')
    A = np.array([[.5, .5], [1., -1.]])
    return np.einsum('ij,j...->i...', A, voltage), np.einsum('ij,j...->i...', np.linalg.inv(A).T, current)


def coupled_native_rows_to_mixed(primary_voltage, differential_voltage, total_current, primary_current):
    """Existing coupled contour rows are Vp,Vp-Vn,Ip+In,Ip; retain the SI dual."""
    V=np.stack((np.asarray(primary_voltage),np.asarray(primary_voltage)-np.asarray(differential_voltage)))
    I=np.stack((np.asarray(primary_current),np.asarray(total_current)-np.asarray(primary_current)))
    return mixed_transform(V,I)


def signed_peak_power(voltage, current):
    voltage, current = np.asarray(voltage, dtype=np.complex128), np.asarray(current, dtype=np.complex128)
    if voltage.shape != current.shape:
        raise ValueError('Voltage and current must have identical physical shapes.')
    return .5*np.real(np.sum(voltage*np.conj(current), axis=0))


def physical_transfer(voltage, current):
    """Two stations, n conductors, 2n genuinely independent source columns."""
    voltage, current = np.asarray(voltage, dtype=np.complex128), np.asarray(current, dtype=np.complex128)
    if voltage.shape != current.shape or voltage.ndim != 3 or voltage.shape[0] != 2 or voltage.shape[1] not in (1, 2) or voltage.shape[2] != 2*voltage.shape[1]:
        raise ValueError('A complete physical transfer needs shape (two stations, n conductors, 2n drives).')
    state = np.concatenate((voltage, 50.*current), axis=1)
    if not np.isfinite(state).all():
        raise ValueError('All physical state samples must be finite.')
    condition = [float(np.linalg.cond(X)) for X in state]
    if max(condition) > 1e8:
        raise ValueError('Physical source states do not resolve a well-conditioned complete transfer.')
    transfer = np.linalg.solve(state[0].T, state[1].T).T
    return dict(matrix=transfer, condition_50ohm=condition,
        state=state, signed_drive_power_W=signed_peak_power(voltage.transpose(1,0,2), current.transpose(1,0,2)),
        reconstruction_relative=float(np.linalg.norm(transfer@state[0]-state[1])/np.linalg.norm(state[1])))


def two_length_modes(short, long, *, lengths, frequency, maximum_index=1.5):
    """Extract independent-length propagation, without modal impedance inputs.

    T(L)=right-state/left-state uses a fixed SI functional in each run. For a
    uniform line T(long) inv(T(short)) advances the extra physical distance.
    Signed VI power identifies direction; no absolute value hides negative loss.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.shape != (2,) or not np.isfinite(lengths).all() or not 0 < lengths[0] < lengths[1]:
        raise ValueError('Two distinct increasing physical reference-plane separations are mandatory.')
    distance = float(lengths[1]-lengths[0])
    if not math.isfinite(frequency) or frequency <= 0 or maximum_index < 1 or 2*np.pi*frequency/299792458.*maximum_index*distance >= np.pi:
        raise ValueError('The declared physical index bound must resolve the principal phase branch before seeing fields.')
    A, B = np.asarray(short, dtype=np.complex128), np.asarray(long, dtype=np.complex128)
    if A.shape != B.shape or A.shape not in ((2,2), (4,4)) or not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError('Two complete matching finite physical transfer operators are mandatory.')
    if max(np.linalg.cond(A), np.linalg.cond(B)) > 1e8:
        raise ValueError('Physical length operators are not numerically resolved.')
    Q = np.linalg.solve(A.T, B.T).T
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    n = len(A)//2
    power = signed_peak_power(eigenvectors[:n], eigenvectors[n:]/50.)
    if np.any(abs(power) < 1e-12):
        raise ValueError('A zero-power or degenerate state cannot identify propagation direction.')
    forward = np.flatnonzero(power > 0)
    if len(forward) != n:
        raise ValueError('The measured operator does not resolve exactly n forward and n backward power states.')
    records = []
    for index in forward:
        value = eigenvalues[index]
        vector = eigenvectors[:,index]/math.sqrt(float(power[index]))
        records.append(dict(eigenvalue=value, state_at_one_W=vector,
            signed_power_W=float(signed_peak_power(vector[:n], vector[n:]/50.)),
            voltage_V=vector[:n], current_A=vector[n:]/50.,
            component_V_over_I_ohm=[None if abs(vector[n+j])<=1e-14 else 50.*vector[j]/vector[n+j] for j in range(n)],
            alpha_np_per_m=float(-np.log(abs(value))/distance), beta_rad_per_m=float(np.angle(value)/distance),
            eigensystem_relative=float(np.linalg.norm(Q@vector-value*vector)/np.linalg.norm(Q@vector))))
    records.sort(key=lambda item:item['beta_rad_per_m'])
    forward_vectors=eigenvectors[:,forward]
    current_basis=forward_vectors[n:]/50.
    impedance_matrix=None if np.linalg.cond(current_basis)>1e8 else np.linalg.solve(current_basis.T,forward_vectors[:n].T).T
    return dict(forward_characteristic_impedance_matrix_ohm=impedance_matrix,physical_reference_lengths_m=lengths, independent_length_difference_m=distance,
        quotient=Q, all_eigenvalues=eigenvalues, forward_modes=records,
        commutator_relative=float(np.linalg.norm(B@A-A@B)/(np.linalg.norm(A)*np.linalg.norm(B))),
        scope='Independent lengths only; spatial-station transfer, tail, heat, boundary, mesh and external-reference gates remain separate.')


def validate_pair_contract(short, long):
    """Reject missing/changed physical inputs before combining independent runs."""
    required = {'run_id','separation_m','reference_positions_m','domain','grid','cross_section',
        'materials','frequencies','source_local_stamps','waveform','normalization','source_clock','reference_offset_m'}
    if set(short) != required or set(long) != required:
        raise ValueError('Each length must provide the complete exact physical-input contract.')
    if short['run_id'] == long['run_id']:
        raise ValueError('Two probe pairs from one actual run are not two physical lengths.')
    for key in required-{'run_id','separation_m','reference_positions_m'}:
        if short[key] != long[key]:
            raise ValueError(f'Independent length physical contract differs: {key}.')
    separation = [item['separation_m'] for item in (short,long)]
    if not 0 < separation[0] < separation[1]:
        raise ValueError('Two increasing physical source/termination separations are mandatory.')
    for item in (short,long):
        refs=np.asarray(item['reference_positions_m'],dtype=float)
        if refs.shape!=(2,) or not -item['separation_m']/2 < refs[0] < refs[1] < item['separation_m']/2:
            raise ValueError('Physical references must be strictly inside their own source planes.')
        offset=item['reference_offset_m']
        if not math.isfinite(offset) or offset<=0 or not np.array_equal(refs,[-item['separation_m']/2+offset,item['separation_m']/2-offset]):
            raise ValueError('Physical references must follow the exact same declared source-to-reference offset.')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--length-index',type=int,choices=(0,1),required=True)
    parser.add_argument('--drive-index',type=int,choices=(0,1),required=True)
    parser.add_argument('--steps',type=int,default=1024)
    args=parser.parse_args()
    if args.steps != 1024:
        raise ValueError('This bounded descriptor permits only 1024 functional steps; physical 64 ns needs a separately reviewed launch.')
    report=run_drive(cases()[args.length_index],DRIVE_BASIS[args.drive_index],args.output,
        graph=True,resume=True,planned_steps=1024,stop_step=512)
    if report['actual_field_graph'] is not True or report['actual_observer_graph'] is not True or report['checkpoint']['step']!=512 or report['planned_steps']!=1024:
        raise RuntimeError('The actual functional run did not exercise Graph and the declared checkpoint horizon.')

if __name__=='__main__':main()


def decode(value):
    if isinstance(value,dict):
        if set(value)=={'real','imag'}:return complex(value['real'],value['imag'])
        return {key:decode(item) for key,item in value.items()}
    if isinstance(value,list):return [decode(item) for item in value]
    return value


def review_actual_prefixes(output):
    """Retain every frequency/operator failure of the four actual short runs.

    This does not qualify attenuation. It audits raw file hashes and counters,
    then records the attempted physical transfer rather than counting exit0 as
    a physical pass. Every length has its own two genuinely propagated columns.
    """
    import hashlib
    from scripts.dev.fdtd.printed_line_loss_execution import execution_identity
    output=Path(output);identity=execution_identity();states=[];source_conditions=[];audits=[]
    for length_index,case in enumerate(cases()):
        columns=[];source_columns=[]
        for drive_index,amplitudes in enumerate(DRIVE_BASIS):
            folder=output/f'length-{length_index}-drive-{drive_index}'
            report=decode(json.loads((folder/'execution.json').read_text()))
            if encode(report['case'])!=encode(asdict(case)) or tuple(report['amplitudes'])!=amplitudes:
                raise ValueError('Actual input case, physical reference positions or basis amplitudes changed.')
            if report['source_before']!=identity or report['source_after']!=identity:
                raise ValueError('Actual executed numerical/core/helper identities changed.')
            if report['planned_steps']!=1024 or report['checkpoint']['step']!=512 or report['checkpoint']['total_steps']!=1024:
                raise ValueError('Actual short execution does not retain its complete declared horizon/checkpoint.')
            for name,sha in report['saved_files_sha256'].items():
                if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=sha:raise ValueError('Actual raw file changed after execution.')
            state=np.asarray(report['physical_state'],dtype=np.complex128)
            if state.shape!=(3,2,5) or not np.isfinite(state).all():raise ValueError('Actual full physical state frequency axis is incomplete.')
            np.testing.assert_array_equal(state,np.asarray(report['legacy_native_physical_state'],dtype=np.complex128))
            columns.append(state)
            spectra=report['actual_source_spectra']
            if set(spectra)!={'left_bottom','left_top','right_bottom','right_top'}:raise ValueError('Actual balanced source port set changed.')
            for side in ('left','right'):
                np.testing.assert_array_equal(spectra[side+'_bottom']['thevenin_peak_voltage'],spectra[side+'_top']['thevenin_peak_voltage'])
            source_columns.append([spectra[side+'_bottom']['thevenin_peak_voltage'] for side in ('left','right')])
            audits.append(dict(length_index=length_index,drive_index=drive_index,actual_dt_s=report['actual_dt_s'],actual_duration_s=report['actual_duration_s'],
                declared_source_port_separation_m=case.separation,declared_reference_span_m=case.positions[-1]-case.positions[0],
                actual_field_graph=report['actual_field_graph'],actual_observer_graph=report['actual_observer_graph'],
                physical_full64ns_duration_reached=False,raw_file_sha256=report['saved_files_sha256']))
        states.append(np.stack(columns,axis=-1))
        spectra=np.asarray(source_columns,dtype=np.complex128).transpose(2,1,0)
        source_conditions.append([float(np.linalg.cond(value)) for value in spectra])
    frequency_records=[]
    for f,frequency in enumerate(FREQUENCIES):
        item=dict(frequency_hz=frequency,independent_run_count=4,source_basis_condition_numbers=[values[f] for values in source_conditions])
        try:
            transfers=[]
            for state in states:
                sample=state[[0,2],:,f,:]
                transfers.append(physical_transfer(sample[:,0:1,:],sample[:,1:2,:]))
            item['each_length_transfer']=transfers
            item['two_length_diagnostic']=two_length_modes(transfers[0]['matrix'],transfers[1]['matrix'],
                lengths=tuple(case.positions[-1]-case.positions[0] for case in cases()),frequency=frequency)
            item['operator_extraction_completed']=True
        except ValueError as error:
            item['operator_extraction_completed']=False;item['operator_failure']=str(error)
        item['attenuation_release_qualified']=False
        frequency_records.append(item)
    return dict(schema='actual-two-length-prefix-functional-review/1',actual_field_runs=4,actual_steps_per_run=1024,
        independent_physical_source_separations_m=SEPARATIONS,source_basis_condition_numbers=source_conditions,
        runs=audits,frequency_records=frequency_records,actual_identity=identity,
        physical_release_qualified=False,scope='Short prefix only. Complete64ns, independent tail, heat/flux, boundary, mesh, four-state and external-reference gates remain pending.')
