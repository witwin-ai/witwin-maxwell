"""Physical mixed-mode line extraction from independent native field integrals.

No projected modal impedance or wave normalization enters these operations.
Common V is the mean conductor-to-ground voltage, common I is their sum;
differential V is their difference, differential I is half their difference.
"""
from __future__ import annotations
import numpy as np


def coupled_physical_state(positive_voltage,differential_voltage,total_current,positive_current):
    """Convert four actual conductor path/loop integrals to power-dual pairs."""
    return np.stack((positive_voltage-.5*differential_voltage,differential_voltage,
                     total_current,positive_current-.5*total_current))


def coupled_transfer(voltage,current,positions):
    """Four independent drives identify the full physical four-state transfer."""
    voltage,current=np.asarray(voltage),np.asarray(current)
    if voltage.shape!=current.shape or voltage.shape!=(len(positions),2,4):
        raise ValueError('Coupled native extraction requires [station, common/differential, four independent drives].')
    if len(positions)<2 or any(b<=a for a,b in zip(positions,positions[1:])):
        raise ValueError('Physical stations must be strictly increasing.')
    records=[]
    for left_position,right_position,v0,v1,i0,i1 in zip(positions,positions[1:],voltage,voltage[1:],current,current[1:]):
        left=np.concatenate((v0,50*i0));right=np.concatenate((v1,50*i1))
        transfer=np.linalg.solve(left.T,right.T).T
        eigenvalues,vectors=np.linalg.eig(transfer)
        forward=[i for i,value in enumerate(eigenvalues) if np.angle(value)>0]
        if len(forward)!=2:raise ValueError('Physical transfer has no unambiguous two forward propagating eigenvalues.')
        common=max(forward,key=lambda i:abs(vectors[0,i])/(abs(vectors[1,i])+np.finfo(float).tiny))
        odd=next(i for i in forward if i!=common)
        distance=right_position-left_position
        modes={}
        for name,index,row in (('common',common,0),('differential',odd,1)):
            value,vector=eigenvalues[index],vectors[:,index]
            impedance=50*vector[row]/vector[row+2]
            modes[name]={'characteristic_impedance_ohm':complex(impedance),
                'per_conductor_even_or_odd_impedance_ohm':complex(impedance*(2 if row==0 else .5)),
                'alpha_np_per_m':float(-np.log(abs(value))/distance),
                'beta_rad_per_m':float(np.angle(value)/distance),
                'eigenvalue':complex(value),'physical_eigenvector_scaled':vector}
        records.append({'positions_m':[left_position,right_position],
            'state_condition_number_50ohm':[float(np.linalg.cond(left)),float(np.linalg.cond(right))],
            'physical_transfer_scaled':transfer,'eigenvalues':eigenvalues,'modes':modes})
    return records
