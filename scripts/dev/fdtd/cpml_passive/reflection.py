"""Frequency-domain reflection of the exact discrete harmonic stepping maps.

    Solve the one-sided, PEC-backed layer, with an exact discrete outgoing /
    incoming decomposition in vacuum. This is not a continuum Fresnel formula
    and does not infer reflection from a transient window.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

from model import C0, EPS0, MU0, ETA


def layer_profiles(h, dt, preset="real", theta=0., layers=12, interior=8):
    import torch
    from witwin.maxwell.fdtd.boundary.cpml import _cpml_profile_1d, DEFAULT_CPML_CONFIG
    nodes=np.arange(-interior,layers+1)*h
    cfg=dict(DEFAULT_CPML_CONFIG)
    rate,floor=None,0.
    if preset=="cfs":
        rate=2*np.pi*.3e9
        fraction=np.sin(np.deg2rad(theta))
        floor=min(1.,2*fraction)
        cosine=np.sqrt(1-min(fraction,np.sqrt(1-(4.2/8)**2))**2)
        cfg.update(alpha_max=EPS0*rate,grading_order=2.,reflection=10**(-4.2/cosine))
    result=[]
    for magnetic in (False,True):
        config=dict(cfg)
        if magnetic:
            config["alpha_max"]*=MU0/EPS0
        out=_cpml_profile_1d((nodes[:-1]+nodes[1:])/2 if magnetic else nodes,nodes,
            layers,dt,MU0 if magnetic else EPS0,torch.tensor(MU0/EPS0 if magnetic else 1.,dtype=torch.float64),
            ETA,config,apply_low=False,apply_high=True,shift_rate=rate,shift_floor=floor)
        result.append(tuple(x.numpy() for x in out))
    return nodes,result


def transfer(profile,lam,dt,candidate):
    k,b,c=profile
    if candidate=="current":
        return 1/k+c*lam/(lam-b)
    if candidate=="c3_old":
        return 1/k+c/(lam-b)
    if candidate=="c3_average":
        return 1/k+c*(lam+1)/(2*(lam-b))
    if candidate in ("c1","c2"):
        active=(b>0)&(b<1)
        r,w=np.zeros_like(b),np.zeros_like(b)
        r[active]=-np.log(b[active])/dt
        w[active]=-c[active]/(1-b[active])
        s=2/dt*(lam-1)/(lam+1)
        return 1/k-r*w/(s+r)
    raise ValueError(candidate)


def reflection(frequency,theta,candidate,*,preset="real",evanescent=False,layers=12):
    h=C0/(20*3e9)
    dt=.5*h/(C0*np.sqrt(2))
    nodes,p=layer_profiles(h,dt,preset,theta,layers)
    omega=2*np.pi*frequency
    lam=np.exp(-1j*omega*dt)
    c=1/np.sqrt(EPS0*MU0)
    ky=omega/c*np.sin(np.deg2rad(theta)) if not evanescent else 1.25*omega/c
    k=2/h*np.sin(.5*ky*h)
    temporal=2*np.sin(.5*omega*dt)/(c*dt)
    K2=temporal**2-k*k
    kz=2/h*np.arcsin(np.sqrt(complex(K2))*h/2)
    ge,gh=[transfer(prof,lam,dt,candidate) for prof in p]
    # Interior equations: ge*B*gh*F*u + (Omega^2-k_y^2)*u=0.
    low=ge[1:-1]*gh[:-1]/h**2
    high=ge[1:-1]*gh[1:]/h**2
    diag=-low-high+K2
    count=len(diag)
    band=np.zeros((3,count),complex)
    band[0,1:]=high[:-1]
    band[1]=diag
    band[2,:-1]=low[1:]
    rhs=np.zeros(count,complex)
    rhs[0]=-low[0]  # u_left=1; u_right=0.
    u=solve_banded((1,1),band,rhs,check_finite=False)
    ratio=u[0]
    r=(np.exp(1j*kz*h)-ratio)/(ratio-np.exp(-1j*kz*h))
    # Refer both evanescent and propagating amplitudes to the layer entrance.
    r*=np.exp(2j*kz*nodes[0])
    return float(abs(r))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    rows=[]
    methods=("current","c1","c2","c3_old","c3_average")
    for preset in ("real","cfs","alpha0"):
        for evanescent in (False,True):
            for theta in ((0.,) if evanescent else (0.,35.,60.,80.)):
                for frequency in (.3e9,.6e9,1e9,2e9,3e9):
                    values={c:reflection(frequency,theta,c,preset=preset,evanescent=evanescent) for c in methods}
                    rows.append(dict(preset=preset,evanescent=evanescent,theta=theta,
                        frequency_GHz=frequency*1e-9,polarizations=["TE","TM"],R=values,
                        passed={c:bool(v<=values["current"]*(1+1e-10)+1e-13) for c,v in values.items()}))
    summary={c:dict(max_R=max(r["R"][c] for r in rows),
        failed_rows=sum(not r["passed"][c] for r in rows),
        max_ratio_to_current=max(r["R"][c]/r["R"]["current"] for r in rows)) for c in methods}
    report=dict(method="exact Yee/CN-ADE frequency-domain layer solve; h=c/(20*3 GHz), dt=h/(2*c*sqrt(2)); amplitude R at entrance",
        evanescent="ky=1.25*omega/c; below-cutoff standing waveguide harmonic; R is amplitude ratio, not power",summary=summary,rows=rows)
    print(json.dumps(summary,indent=2),flush=True)
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
