"""Probe: staircased lossy-metal cylinder vs resolved volumetric conductor.

Observable = net power ABSORBED by the cylinder = -(net outward Poynting flux
through a closed box enclosing the cylinder, source outside the box). This
isolates the conductor loss: a PEC cylinder absorbs ~0, a good conductor absorbs
a finite power. SIBC (coarse, surface) vs resolved volumetric (fine, skin-depth
resolved) both measure the SAME physical absorbed power through the SAME box.
"""
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


def skin_depth(sigma, f):
    return math.sqrt(2.0 / (2.0 * math.pi * f * MU0 * sigma))


def _flux_box(freq):
    f = (freq,)
    return [
        mw.FinitePlaneMonitor("xp", position=(BOX, 0, 0), size=(0, 2*BOX, 2*BOX),
                              fields=("Ey", "Ez", "Hy", "Hz"), frequencies=f, compute_flux=True, normal_direction="+"),
        mw.FinitePlaneMonitor("xn", position=(-BOX, 0, 0), size=(0, 2*BOX, 2*BOX),
                              fields=("Ey", "Ez", "Hy", "Hz"), frequencies=f, compute_flux=True, normal_direction="-"),
        mw.FinitePlaneMonitor("yp", position=(0, BOX, 0), size=(2*BOX, 0, 2*BOX),
                              fields=("Ex", "Ez", "Hx", "Hz"), frequencies=f, compute_flux=True, normal_direction="+"),
        mw.FinitePlaneMonitor("yn", position=(0, -BOX, 0), size=(2*BOX, 0, 2*BOX),
                              fields=("Ex", "Ez", "Hx", "Hz"), frequencies=f, compute_flux=True, normal_direction="-"),
        mw.FinitePlaneMonitor("zp", position=(0, 0, BOX), size=(2*BOX, 2*BOX, 0),
                              fields=("Ex", "Ey", "Hx", "Hy"), frequencies=f, compute_flux=True, normal_direction="+"),
        mw.FinitePlaneMonitor("zn", position=(0, 0, -BOX), size=(2*BOX, 2*BOX, 0),
                              fields=("Ex", "Ey", "Hx", "Hy"), frequencies=f, compute_flux=True, normal_direction="-"),
    ]


def absorbed(dx, kind):
    structures = []
    if kind == "sibc":
        structures = [mw.Structure(geometry=Cylinder(radius=R, height=2*HALF_H, axis="z", position=(0, 0, 0)),
                                   material=mw.LossyMetalMedium(conductivity=SIGMA))]
    elif kind == "resolved":
        structures = [mw.Structure(geometry=Cylinder(radius=R, height=2*HALF_H, axis="z", position=(0, 0, 0)),
                                   material=mw.Material(eps_r=1.0, sigma_e=SIGMA))]
    elif kind == "pec":
        structures = [mw.Structure(geometry=Cylinder(radius=R, height=2*HALF_H, axis="z", position=(0, 0, 0)),
                                   material=mw.Material.pec())]
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-DOMAIN, DOMAIN), (-DOMAIN, DOMAIN), (-DOMAIN, DOMAIN))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=8, strength=1.0),
        device="cuda",
        sources=[mw.PointDipole(position=(-0.030, 0.0, 0.0), polarization=(0, 0, 1.0),
                                width=SRC_W, source_time=mw.CW(frequency=F, amplitude=40.0), name="s")],
        structures=structures,
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
    return -tot  # net inward = absorbed


print(f"skin depth delta={skin_depth(SIGMA, F)*1e3:.3f} mm  R/delta={R/skin_depth(SIGMA, F):.2f}")
a_pec = absorbed(0.0015, "pec")
a_sibc = absorbed(0.0015, "sibc")
a_res = {}
for dx in (0.00075, 0.0005):
    a_res[dx] = absorbed(dx, "resolved")
fine = a_res[0.0005]
print(f"absorbed PEC   (coarse) = {a_pec:.4e}")
print(f"absorbed SIBC  (coarse) = {a_sibc:.4e}")
print(f"absorbed resolved 0.75mm= {a_res[0.00075]:.4e}")
print(f"absorbed resolved 0.50mm= {fine:.4e}")
print(f"SIBC vs resolved-fine  rel = {abs(a_sibc-fine)/abs(fine):.4f}")
print(f"resolved tier 0.75->0.5 rel = {abs(a_res[0.00075]-fine)/abs(fine):.4f}")
print(f"PEC/lossy absorbed ratio = {a_pec/a_sibc:.4f}  (falsification: PEC absorbs ~0)")
