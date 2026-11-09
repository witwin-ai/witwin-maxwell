"""Single-pi fit and reference comparisons for the F-A substrate/shield study.

These CPU fp64 analysis routines never enter the solver runtime. The effective
shunt loss also includes resistive shield dissipation; a fit alone cannot
separate shield heating from silicon heating.
"""
import hashlib
import json

import numpy as np
from scipy.optimize import least_squares
from tests.support.rfic_fa_references import extract_fa_admittance,resonance_and_peak


def pi_admittance(frequencies, parameters):
    """exp(-i omega t): series (Rs-i omega Ls) parallel Cs; two substrate arms.

    Each arm is Cox in series with Rsub parallel Csub. Parameters are SI:
    Ls, Rs, Cs, Cox1, Csub1, Rsub1, Cox2, Csub2, Rsub2.
    """
    omega = 2*np.pi*np.asarray(frequencies)
    ls,rs,cs,*arms = parameters
    branch = 1/(rs-1j*omega*ls)-1j*omega*cs
    y = np.zeros((len(omega),2,2),dtype=complex)
    y[:,0,1] = y[:,1,0] = -branch
    for side in (0,1):
        cox,csub,rsub = arms[3*side:3*side+3]
        substrate = 1/rsub-1j*omega*csub
        arm = 1/(1/(-1j*omega*cox)+1/substrate)
        y[:,side,side] = branch+arm
    return y


def pi_terms(y):
    return np.stack((-(y[:,0,1]+y[:,1,0])/2,
                     y[:,0,0]+y[:,0,1], y[:,1,1]+y[:,1,0]),-1)


def fit_pi(frequencies, y, *, fit_band=(.5e9,8e9)):
    """Fit positive component values, reporting each complex-branch error.

    Branches receive separate norm weights so large series current cannot hide
    substrate-loss errors. A constant Rs cannot reproduce all skin/proximity
    dispersion; the measured residual quantifies that model limitation.
    """
    frequencies,y = np.asarray(frequencies),np.asarray(y)
    select = (frequencies>=fit_band[0])&(frequencies<=fit_band[1])
    f,target = frequencies[select],pi_terms(y[select])
    scale = np.sqrt(np.mean(np.abs(target)**2,axis=0))
    initial = np.array([3.3e-9,3.,5e-15,80e-15,30e-15,1000.,80e-15,30e-15,1000.])
    lower = np.array([.01e-9,.001,.001e-15,*([.001e-15,.001e-15,.001]*2)])
    upper = np.array([100e-9,1e4,1e-10,*([1e-10,1e-10,1e10]*2)])
    def residual(log_parameters):
        error = (pi_terms(pi_admittance(f,np.exp(log_parameters)))-target)/scale
        return np.concatenate((error.real.ravel(),error.imag.ravel()))
    fit = least_squares(residual,np.log(initial),bounds=(np.log(lower),np.log(upper)),
                        xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=10000)
    parameters = np.exp(fit.x)
    modeled = pi_admittance(frequencies,parameters)
    error = pi_terms(modeled[select])-target
    relative = np.sqrt(np.mean(np.abs(error)**2,axis=0))/scale
    names = ("Ls_H","Rs_ohm","Cs_F","Cox1_F","Csub1_F","Rsub1_ohm",
             "Cox2_F","Csub2_F","Rsub2_ohm")
    return dict(parameters=dict(zip(names,parameters.tolist())),fit_band_Hz=list(fit_band),
                relative_branch_rms=relative.tolist(),
                relative_branch_real_rms=(np.sqrt(np.mean(error.real**2,axis=0))/
                                          np.sqrt(np.mean(target.real**2,axis=0))).tolist(),
                relative_branch_imag_rms=(np.sqrt(np.mean(error.imag**2,axis=0))/
                                          np.sqrt(np.mean(target.imag**2,axis=0))).tolist(),
                sampled_fit_band_Hz=[float(f[0]),float(f[-1])],
                success=bool(fit.success),
                evaluations=fit.nfev,active_bound_parameters=[name for name,active in zip(names,fit.active_mask) if active],
                jacobian_condition_number=float(np.linalg.cond(fit.jac)),
                model=metrics(frequencies,modeled),
                shunt_conductance_S=pi_terms(modeled)[:,1:].real.tolist())


def metrics(frequencies,y):
    frequencies,y = np.asarray(frequencies),np.asarray(y)
    data=extract_fa_admittance(frequencies,y)
    return dict(**data,**resonance_and_peak(data),
                L_1GHz_H=(float(np.interp(1e9,frequencies,data["L_H"]))
                            if frequencies[0]<=1e9<=frequencies[-1] else None))

def unpack_network(data):
    return np.array(data["y_real"])+1j*np.array(data["y_imag"])


def pi_attribution(frequencies, baseline_fit, shield_fit):
    """Counterfactual series/shunt swaps quantify the fitted Q change.

    This is a circuit-model explanation, not a unique electromagnetic loss
    partition: the fitted Rsub includes shield heating when a shield is present.
    """
    base = list(baseline_fit["parameters"].values())
    shield = list(shield_fit["parameters"].values())
    combinations = dict(baseline=base,shield=shield,
                        only_shunt_changed=base[:3]+shield[3:],
                        only_series_changed=shield[:3]+base[3:])
    for name,indices in (("only_Cox_changed",(3,6)),("only_Csub_changed",(4,7)),
                         ("only_Rsub_changed",(5,8))):
        values=base.copy()
        for index in indices:
            values[index]=shield[index]
        combinations[name]=values
    return {name: {k: value for k,value in metrics(frequencies,pi_admittance(frequencies,values)).items()
                   if k in ("Q_peak","peak_frequency_Hz","self_resonance_Hz","L_1GHz_H")}
            for name,values in combinations.items()}


def geometry_digest(rectangles, materials, vias, via_materials):
    """Hash nominal physical layout/material records, canonicalized to 1 pm.

    The fixture coordinates lie on an integer/half-micrometre layout lattice.
    Canonicalization removes only SI/um conversion roundoff, including Box
    bounds. Ideal source bridges are recorded separately from physical metal.
    """
    records=[]
    for kind,shapes,media in (("sheet",rectangles,materials),("via",vias,via_materials)):
        for shape,(sigma,thickness,roughness) in zip(shapes,media):
            records.append((kind,sigma,round(thickness*1e6,6),roughness,
                            *(round(value*1e6,6) for value in shape)))
    return hashlib.sha256(json.dumps(sorted(records),separators=(",",":")).encode()).hexdigest()
