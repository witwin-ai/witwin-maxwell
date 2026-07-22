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
R = 0.008
HALF_H = 0.008
DOMAIN = 0.045
SRC_W = 0.003
BOX = 0.014


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
        sources=[mw.PointDipole(position=(-0.030, 0, 0), polarization=(0, 0, 1.0), width=SRC_W,
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


fine = absorbed(0.0005, "resolved")
print(f"resolved 0.5mm = {fine:.4e}")
for dx in (0.0015, 0.001, 0.00075):
    a = absorbed(dx, "sibc")
    print(f"SIBC dx={dx*1e3:.2f}mm absorbed={a:.4e}  rel vs resolved-fine = {abs(a-fine)/fine:.4f}")
