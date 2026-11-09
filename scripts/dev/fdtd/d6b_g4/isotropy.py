"""D6b G4: in-plane rotation isotropy of the conformal thin-sheet runtime (no port).

A half-wave strip dipole lies on a flat conformal sheet (normal z, offset from the
Yee planes) and is illuminated at normal incidence with E along the strip axis.
The strip and the polarization are rotated in-plane by phi; an on-axis observer
on the source side records the scattered field projected on the strip axis.
Rotation invariance means the scattering spectrum, hence its resonance, does not
depend on phi.  Gate: |f_res(phi) / f_res(0) - 1| <= 0.5%.

Usage: python isotropy.py OUT_JSON [--dl 0.005] [--materials pec,cu] [--angles 0,15,30,45]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import witwin.maxwell as mw
from tests.materials.thin_sheet.harness import rectangle_mesh, field_series, pec_material

C0 = 299792458.0
F0 = 1.0e9


def scene_for(phi_deg, material, dl, *, with_sheet, offset=(0.0, 0.0), native=False, zoff=0.3):
    wavelength = C0 / F0
    length, width = 0.47 * wavelength, 0.015
    half = math.ceil((0.5 * length + 0.5 * wavelength) / dl) * dl
    zhalf = math.ceil(0.5 * wavelength / dl) * dl
    z_sheet = zoff * dl
    phi = math.radians(phi_deg)
    axis = np.array((math.cos(phi), math.sin(phi), 0.0))
    structures = []
    if with_sheet:
        center = (offset[0] * dl, offset[1] * dl, z_sheet)
        if native == "planar":
            # PlanarPEC: zero-volume PEC polygon on the z = 0 Yee node plane (edge ownership).
            c, r = np.array(center[:2]), np.array((-axis[1], axis[0]))
            a, b = 0.5 * length * axis[:2], 0.5 * width * r
            corners = [tuple(c - a - b), tuple(c + a - b), tuple(c + a + b), tuple(c - a + b)]
            geometry = mw.PlanarPEC(corners, normal_axis="z", coordinate=0.0)
        elif native:
            # Validated axis-aligned planar foil path on the z = 0 grid plane (0 deg only).
            from witwin.core import Box
            assert phi_deg == 0.0
            geometry = Box(position=(center[0], center[1], 0.0), size=(length, width, 0.0))
        else:
            geometry = rectangle_mesh(center, (0.0, 0.0, 1.0), 0.5 * length, 0.5 * width,
                                      u_direction=axis, sheet_model="conformal_two_sided")
        structures.append(mw.Structure(geometry=geometry, material=material))
    observer = (0.0, 0.0, round((z_sheet + 0.4 * wavelength) / dl) * dl)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-zhalf, zhalf))),
        grid=mw.GridSpec.uniform(dl),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=10), device="cuda",
        sources=[mw.PlaneWave(direction=(0.0, 0.0, -1.0), polarization=tuple(axis),
                              source_time=mw.GaussianPulse(frequency=F0, fwidth=0.5 * F0))],
        structures=structures,
        monitors=[mw.FieldTimeMonitor("obs", components=("Ex", "Ey"), position=observer)])
    return scene, axis


def projected_series(scene, axis, steps, dt):
    """Run with an explicit timestep so empty and loaded runs share one clock."""
    result = mw.Simulation.fdtd(scene, frequencies=[F0], run_time=mw.TimeConfig(time_steps=steps, dt=dt),
                                backend="native").run()
    t, ex = field_series(result, "obs", "Ex")
    _, ey = field_series(result, "obs", "Ey")
    return t, axis[0] * ex + axis[1] * ey


def native_dt(scene):
    """The solver's own (cut-bound-limited) timestep for this scene."""
    return float(mw.Simulation.fdtd(scene, frequencies=[F0], run_time=mw.TimeConfig(time_steps=1),
                                    backend="native").prepare().solver.dt)


def resonance(t, scattered, incident):
    freqs = np.linspace(0.6 * F0, 1.4 * F0, 1601)
    kernel = np.exp(2j * np.pi * np.outer(freqs, t))
    response = np.abs(kernel @ scattered) / np.abs(kernel @ incident)
    k = int(np.argmax(response))
    y0, y1, y2 = response[k - 1:k + 2]
    shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    step = freqs[1] - freqs[0]
    return float(freqs[k] + shift * step), float(y1), response


def pencil_pole(t, signal, start, *, order=8):
    """Dominant natural resonance from late-time ringing (matrix pencil).

    Returns the complex frequency (f, decay) of the mode with the largest
    residue among poles inside the source band.
    """
    dt = float(t[1] - t[0])
    stride = max(1, int(round(1.0 / (8 * 1.4 * F0 * dt))))
    y = signal[t >= start][::stride]
    n = len(y)
    rows = n // 2
    H = np.array([y[i:i + rows + 1] for i in range(n - rows)])
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    V = Vh[:order].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    s = np.log(z) / (stride * dt)
    freq = s.imag / (2 * np.pi)
    basis = z[None, :] ** np.arange(n)[:, None]
    residue = np.linalg.lstsq(basis, y.astype(complex), rcond=None)[0]
    keep = (freq > 0.2 * F0) & (freq < 1.8 * F0)
    k = np.flatnonzero(keep)[int(np.argmax(np.abs(residue[keep])))]
    return float(freq[k]), float(-s[k].real)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--dl", type=float, default=0.005)
    parser.add_argument("--materials", default="pec,cu")
    parser.add_argument("--angles", default="0,15,30,45")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--offset", default="0,0", help="strip centre offset in cells (x,y)")
    parser.add_argument("--zoff", type=float, default=0.3, help="sheet height above the z = 0 Yee plane, in cells")
    parser.add_argument("--native", nargs="?", const=True, default=False,
                        help="native axis-aligned foil reference (0 deg); 'planar' for PlanarPEC (any angle)")
    args = parser.parse_args()
    materials = {"pec": pec_material(),
                 "cu": mw.CopperFoil(thickness=35e-6, conductivity=5.8e7, frequency_range=(0.5e9, 1.5e9))}
    angles = [float(a) for a in args.angles.split(",")]
    report = {"dl": args.dl, "steps": args.steps, "f0": F0, "cases": {}}
    for name in args.materials.split(","):
        rows = {}
        for phi in angles:
            start = time.time()
            empty, axis = scene_for(phi, materials[name], args.dl, with_sheet=False, zoff=args.zoff)
            offset = tuple(float(v) for v in args.offset.split(","))
            loaded, _ = scene_for(phi, materials[name], args.dl, with_sheet=True, offset=offset, native=args.native, zoff=args.zoff)
            # One shared clock: the loaded scene's cut bound limits dt; the physical
            # duration is fixed by --steps times the empty-scene dt.
            dt_empty, dt_loaded = native_dt(empty), native_dt(loaded)
            dt = 0.999 * min(dt_empty, dt_loaded)
            steps = int(math.ceil(args.steps * dt_empty / dt))
            t, incident = projected_series(empty, axis, steps, dt)
            t_loaded, total = projected_series(loaded, axis, steps, dt)
            assert np.array_equal(t, t_loaded), "empty and loaded clocks differ"
            scattered = total - incident
            f_peak, peak, _ = resonance(t, scattered, incident)
            # Late time: after the incident pulse (centre ~ a few pulse widths) has
            # passed the strip and the observer twice.
            late = float(t[int(np.argmax(np.abs(incident)))]) + 6.0 / F0
            f_res, decay = pencil_pole(t, scattered, late)
            np.savez(args.out.replace(".json", f"-{name}-{phi:g}.npz"), t=t, scattered=scattered, incident=incident)
            rows[phi] = {"f_res": f_res, "decay": decay, "f_peak": f_peak, "peak": peak,
                         "dt": dt, "dt_empty": dt_empty, "dt_loaded": dt_loaded, "steps": steps,
                         "seconds": time.time() - start,
                         "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
            print(name, phi, rows[phi], flush=True)
        base = rows[angles[0]]["f_res"]
        for phi, row in rows.items():
            row["shift"] = row["f_res"] / base - 1
            row["peak_ratio"] = row["peak"] / rows[angles[0]]["peak"]
        report["cases"][name] = rows
        json.dump(report, open(args.out, "w"), indent=2)
    worst = max(abs(r["shift"]) for rows in report["cases"].values() for r in rows.values())
    report["worst_shift"] = worst
    report["gate_pass"] = worst <= 5e-3
    json.dump(report, open(args.out, "w"), indent=2)
    print("WORST", worst, "PASS" if worst <= 5e-3 else "FAIL", flush=True)


if __name__ == "__main__":
    main()
