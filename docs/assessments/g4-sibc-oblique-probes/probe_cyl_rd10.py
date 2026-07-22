"""SIBC absorbed-power convergence across grid tiers vs resolved reference."""
from __future__ import annotations
import math
import numpy as np
import torch
import witwin.maxwell as mw
from witwin.core import Cylinder

MU0 = 4.0e-7 * math.pi
F = 6.0e9
SIGMA = 30.0
R = 0.012
HALF_H = 0.012
DOMAIN = 0.055
SRC_W = 0.003
BOX = 0.020


def _flux_box(freq):
    f = (freq,)
    mk = mw.FinitePlaneMonitor
    return [
        mk("xp", position=(BOX, 0, 0), size=(0, 2*BOX, 2*BOX), fields=("Ey", "Ez", "Hy", "Hz"), frequencies=f, compute_flux=True, normal_direction="+"),
        mk("xn", position=(-BOX, 0, 0), size=(0, 2*BOX, 2*BOX), fields=("Ey", "Ez", "Hy", "Hz"), frequencies=f, compute_flux=True, normal_direction="-"),
        mk("yp", position=(0, BOX, 0), size=(2*BOX, 0, 2*BOX), fields=("Ex", "Ez", "Hx", "Hz"), frequencies=f, compute_flux=True, normal_direction="+"),
        mk("yn", position=(0, -BOX, 0), size=(2*BOX, 0, 2*BOX), fields=("Ex", "Ez", "Hx", "Hz"), frequencies=f, compute_flux=True, normal_direction="-"),
        mk("zp", position=(0, 0, BOX), size=(2*BOX, 2*BOX, 0), fields=("Ex", "Ey", "Hx", "Hy"), frequencies=f, compute_flux=True, normal_direction="+"),
        mk("zn", position=(0, 0, -BOX), size=(2*BOX, 2*BOX, 0), fields=("Ex", "Ey", "Hx", "Hy"), frequencies=f, compute_flux=True, normal_direction="-"),
    ]


def absorbed(dx, kind):
    if kind == "sibc":
        mat = mw.LossyMetalMedium(conductivity=SIGMA)
    else:
        mat = mw.Material(eps_r=1.0, sigma_e=SIGMA)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-DOMAIN, DOMAIN),)*3),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=8, strength=1.0),
        device="cuda",
        sources=[mw.PointDipole(position=(-0.040, 0, 0), polarization=(0, 0, 1.0), width=SRC_W,
                                source_time=mw.CW(frequency=F, amplitude=40.0), name="s")],
        structures=[mw.Structure(geometry=Cylinder(radius=R, height=2*HALF_H, axis="z", position=(0, 0, 0)), material=mat)],
        monitors=_flux_box(F),
    )
    result = mw.Simulation.fdtd(scene, frequencies=[F],
                                run_time=mw.TimeConfig.auto(steady_cycles=14, transient_cycles=20),
                                full_field_dft=False).run()
    tot = 0.0
    for nm in ("xp", "xn", "yp", "yn", "zp", "zn"):
        fv = result.monitor(nm)["flux"]
        fv = fv.detach().cpu().numpy() if hasattr(fv, "detach") else np.asarray(fv)
        tot += float(fv.reshape(-1)[0])
    del result, scene
    torch.cuda.empty_cache()
    return -tot


import math as _m
delta=_m.sqrt(2.0/(2.0*_m.pi*F*MU0*SIGMA))
print(f"R/delta={R/delta:.2f}")
fine = absorbed(0.0005, "resolved")
print(f"resolved 0.5mm = {fine:.4e}")
a = absorbed(0.001, "sibc")
print(f"SIBC dx=1.00mm absorbed={a:.4e}  rel vs resolved-fine = {abs(a-fine)/fine:.4f}")
# PEC falsification
def absorbed_pec(dx):
    import witwin.maxwell as _mw
    from witwin.core import Cylinder as _Cyl
    scene=_mw.Scene(domain=_mw.Domain(bounds=((-DOMAIN,DOMAIN),)*3),grid=_mw.GridSpec.uniform(dx),
        boundary=_mw.BoundarySpec.faces(default="pml",num_layers=8,strength=1.0),device="cuda",
        sources=[_mw.PointDipole(position=(-0.040,0,0),polarization=(0,0,1.0),width=SRC_W,source_time=_mw.CW(frequency=F,amplitude=40.0),name="s")],
        structures=[_mw.Structure(geometry=_Cyl(radius=R,height=2*HALF_H,axis="z",position=(0,0,0)),material=_mw.Material.pec())],
        monitors=_flux_box(F))
    r=_mw.Simulation.fdtd(scene,frequencies=[F],run_time=_mw.TimeConfig.auto(steady_cycles=14,transient_cycles=20),full_field_dft=False).run()
    t=0.0
    for nm in ("xp","xn","yp","yn","zp","zn"):
        fv=r.monitor(nm)["flux"];import numpy as _np
        fv=fv.detach().cpu().numpy() if hasattr(fv,"detach") else _np.asarray(fv); t+=float(fv.reshape(-1)[0])
    import torch as _t; del r,scene; _t.cuda.empty_cache(); return -t
ap=absorbed_pec(0.001)
print(f"PEC absorbed={ap:.4e}  ratio PEC/SIBC={ap/a:.4f}")
