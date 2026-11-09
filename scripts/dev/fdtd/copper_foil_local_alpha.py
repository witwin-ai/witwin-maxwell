"""Local CopperFoil attenuation alpha_c = P'_loss / (2 P) on short printed lines.

A CW current source launches the quasi-TEM mode of a short x line (PML at both
x ends, sigma = 0 so the foil is the only loss). At every station downstream,
P' is the native foil heat per unit length (``surface_power_traces``) and P the
Poynting flux, both restricted to the physical cross-section. H is the exact
discrete Faraday image of the E phasors, using each foil's high-face trace for
the H row above it, so no observer interpolation enters. Per-face currents are
also checked against Rs*|J|^2 of the same H (``--faces``).

Kinds: ``plate`` (parallel plates d = 10 mm, periodic y: no edges, analytic
Rs/(eta d)); ``stripline`` (W = 10 mm, b = 10 mm, PEC side walls at +-30 mm);
``cpw`` and ``pair`` (the public-line stack-ups and +-0.14/+-0.13 m box, 1.25 mm
inside +-20 mm and 2.5 mm outside; ``--mode odd|even`` for the pair).

    python -m scripts.dev.fdtd.copper_foil_local_alpha --kind stripline --h 0.00125 --edge-model none
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import witwin.maxwell as mw

MU0 = 1.25663706212e-6
EPS0 = 8.8541878128e-12


def _graded(half, fine, coarse, radius=.02):
    inner = np.arange(-round(radius / fine), round(radius / fine) + 1) * fine
    outer = np.arange(-round(half / coarse), round(half / coarse) + 1) * coarse
    return np.r_[outer[outer < -radius - 1e-12], inner, outer[outer > radius + 1e-12]]


def build(kind, *, h, frequency, edge_model, mode="odd", half_x=.06):
    big = 10.
    foil = dict(thickness=35e-6, frequency_range=(.8e9, 1.2e9), edge_model=edge_model)
    if kind in ("plate", "stripline"):
        xs = np.arange(-round(half_x / h) - 10, round(half_x / h) + 11) * h
        ys = np.arange(-round(.03 / h), round(.03 / h) + 1) * h
        zs = np.arange(-round(.0075 / h), round(.0075 / h) + 1) * h
        step = h
    else:
        step = .0025
        xs = np.arange(-round(half_x / step), round(half_x / step) + 1) * step
        ys, zs = _graded(.14, h, step), _graded(.13, h, step)
    structures = [mw.Structure(mw.Box(size=(big,) * 3), mw.Material(eps_r=2.25), name="dielectric")
                  if kind != "pair" else
                  mw.Structure(mw.Box(position=(0., 0., -.0025), size=(big, big, .005)), mw.Material(eps_r=2.25),
                               name="substrate")]

    def sheet(name, y0, y1, z):
        structures.append(mw.Structure(mw.Box(position=torch.tensor((0., (y0 + y1) / 2, z), dtype=torch.float64),
            size=torch.tensor((big, y1 - y0, 0.), dtype=torch.float64)), mw.CopperFoil(**foil), name=name))

    x0 = -round(.6 * half_x / step) * step
    sources = []
    if kind in ("plate", "stripline"):
        sheet("ground_low", -big / 2, big / 2, -.005)
        sheet("ground_high", -big / 2, big / 2, .005)
        faces = dict(x=("pml", "pml"), y=("periodic", "periodic")) if kind == "plate" else dict(x=("pml", "pml"))
        boundary = mw.BoundarySpec.faces(default="pec", num_layers=10, strength=1e6, **faces)
        if kind == "plate":
            sources.append(mw.UniformCurrentSource(size=(h / 2, .06, .010), polarization="Ez", center=(x0, 0., 0.),
                                                   source_time=mw.CW(frequency)))
        else:
            sheet("signal", -.005, .005, 0.)
            sources += [mw.UniformCurrentSource(size=(h / 2, .010, .005), polarization=(0., 0., sign),
                        center=(x0, 0., sign * .0025), source_time=mw.CW(frequency)) for sign in (1., -1.)]
    elif kind == "cpw":
        sheet("ground_left", -big / 2, -.010, 0.)
        sheet("ground_right", .010, big / 2, 0.)
        sheet("signal", -.005, .005, 0.)
        sources += [mw.UniformCurrentSource(size=(step / 2, .005, h / 2), polarization=(0., sign, 0.),
                    center=(x0, sign * .0075, 0.), source_time=mw.CW(frequency)) for sign in (1., -1.)]
        boundary = mw.BoundarySpec.pml(num_layers=8)
    else:
        sheet("ground", -big / 2, big / 2, -.005)
        sheet("signal_positive", .0025, .0075, 0.)
        sheet("signal_negative", -.0075, -.0025, 0.)
        signs = (-1., -1.) if mode == "even" else (-1., 1.)
        sources += [mw.UniformCurrentSource(size=(step / 2, .005, .005), polarization=(0., 0., sign),
                    center=(x0, y, -.0025), source_time=mw.CW(frequency)) for sign, y in zip(signs, (.005, -.005))]
        boundary = mw.BoundarySpec.pml(num_layers=8)
    scene = mw.Scene(domain=mw.Domain(bounds=((xs[0], xs[-1]), (ys[0], ys[-1]), (zs[0], zs[-1]))),
                     grid=mw.GridSpec.custom(xs, ys, zs), boundary=boundary, device="cuda", structures=structures)
    for source in sources:
        scene.add_source(source)
    return scene


def _foil_phasors(solver):
    from witwin.maxwell.fdtd.runtime.representation import spectral_phasor_factor
    for entry in solver._copper_foils:
        if entry["imaginary"]:
            continue
        phasor = torch.complex(entry["spectral_real"], entry["spectral_imag"]) * spectral_phasor_factor(solver)
        yield entry, (phasor / entry["spectral_norm"].clamp_min(1)[:, None, None])[0].cpu().numpy()


def faraday_h(result, frequency):
    """Hy, Hz phasors from the discrete Faraday law of the sampled E (z-normal foils)."""
    solver = result.solver
    xs, ys, zs = (np.asarray(getattr(solver.scene, f"{axis}_nodes64")) for axis in "xyz")
    E = {name: result.tensor(name).cpu().numpy().astype(np.complex128) for name in ("Ex", "Ey", "Ez")}
    dt = float(solver.dt)
    factor = 1 / (1j * MU0 * 2 * math.sin(math.pi * frequency * dt) / dt)
    dz = np.diff(zs)
    dexdz = (E["Ex"][:, :, 1:] - E["Ex"][:, :, :-1]) / dz[None, None, :]
    ex_plane, ey_plane = E["Ex"].copy(), E["Ey"].copy()
    for entry, (low, high) in _foil_phasors(solver):
        index = tuple(entry["indices"].cpu().numpy().T)
        weight = entry["right_width"] / (entry["left_width"] + entry["right_width"])
        if entry["electric_name"] == "Ex":
            dexdz[index[0], index[1], entry["face"].normal_index] -= (high - low) / dz[entry["face"].normal_index]
            ex_plane[index] = low + (high - low) * weight
        else:
            ey_plane[index] = low + (high - low) * weight
    hy = factor * (dexdz - (E["Ez"][1:] - E["Ez"][:-1]) / np.diff(xs)[:, None, None])
    hz = factor * ((ey_plane[1:] - ey_plane[:-1]) / np.diff(xs)[:, None, None]
                   - (ex_plane[:, 1:] - ex_plane[:, :-1]) / np.diff(ys)[None, :, None])
    return E, hy, hz, (xs, ys, zs)


def local_alpha(result, frequency, *, half_y, half_z):
    E, hy, hz, (xs, ys, zs) = faraday_h(result, frequency)
    solver = result.solver
    dy, dz = np.diff(ys), np.diff(zs)
    dyd = np.r_[dy[0] / 2, (dy[:-1] + dy[1:]) / 2, dy[-1] / 2]
    dzd = np.r_[dz[0] / 2, (dz[:-1] + dz[1:]) / 2, dz[-1] / 2]
    if solver.scene.boundary.axis_kind("y") == "periodic":
        dyd[0] = dyd[-1] = dy[0] / 2
    inside_y, inside_z = np.abs(ys) < half_y, np.abs(zs) < half_z
    mid_y, mid_z = np.abs((ys[1:] + ys[:-1]) / 2) < half_y, np.abs((zs[1:] + zs[:-1]) / 2) < half_z
    traces = [trace for record in result.conductor_losses.values() for trace in record["surface_power_traces"]]
    pml = solver.scene.pml_thickness_for_face("x", "low")
    rows = []
    for station in range(len(xs) // 2 - 4, len(xs) - pml - 6):
        flux = .5 * np.real(
            np.sum(E["Ey"][station] * np.conj((hz[station - 1] + hz[station]) / 2) * (dy * mid_y)[:, None] * (dzd * inside_z)[None])
            - np.sum(E["Ez"][station] * np.conj((hy[station - 1] + hy[station]) / 2) * (dyd * inside_y)[:, None] * (dz * mid_z)[None]))
        lower = xs[station] - (xs[station] - xs[station - 1]) / 2
        upper = xs[station] + (xs[station + 1] - xs[station]) / 2
        heat = rim = 0.
        for trace in traces:
            low, high = trace["lower_bounds"].numpy(), trace["upper_bounds"].numpy()
            inside = (np.abs(low[:, 1]) < half_y) & (np.abs(low[:, 2]) < half_z)
            fraction = np.clip(np.minimum(high[:, 0], upper) - np.maximum(low[:, 0], lower), 0, None) / (high[:, 0] - low[:, 0])
            power = trace["power"][0].numpy() * fraction * inside
            heat += power.sum()
            rim += power[trace["edge_impedance_factor"].numpy() != 1].sum()
        rows.append(dict(x_m=float(xs[station]), heat_per_m=heat / (upper - lower), flux_W=flux,
                         alpha_np_per_m=heat / (upper - lower) / (2 * flux), edge_trace_share=rim / heat))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("plate", "stripline", "cpw", "pair"), required=True)
    parser.add_argument("--mode", choices=("odd", "even"), default="odd")
    parser.add_argument("--h", type=float, default=.00125)
    parser.add_argument("--frequency", type=float, default=1e9)
    parser.add_argument("--edge-model", choices=mw.CopperFoil.EDGE_MODELS, default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    scene = build(args.kind, h=args.h, frequency=args.frequency, edge_model=args.edge_model, mode=args.mode)
    # About 15 ns transient plus a ten-period Hanning window at 1 GHz.
    steps = args.steps or round((6000 * .0025 if args.kind in ("plate", "stripline") else 9000 * .00125) / args.h)
    start = time.perf_counter()
    result = mw.Simulation.fdtd(scene, frequencies=[args.frequency], full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=steps), spectral_sampler=mw.SpectralSampler(window="hanning")).run()
    box = (.03 + 1e-9, .0075 + 1e-9) if args.kind in ("plate", "stripline") else (.14 + 1e-9, .13 + 1e-9)
    rows = local_alpha(result, args.frequency, half_y=box[0], half_z=box[1])
    alpha = np.array([row["alpha_np_per_m"] for row in rows])
    report = dict(kind=args.kind, mode=args.mode, h_m=args.h, frequency_hz=args.frequency, edge_model=args.edge_model,
                  steps=steps, alpha_np_per_m=float(alpha.mean()), alpha_station_std=float(alpha.std()),
                  edge_trace_share=float(np.mean([row["edge_trace_share"] for row in rows])),
                  free_edge_trace_count={name: record["free_edge_trace_count"]
                                         for name, record in result.conductor_losses.items()},
                  wall_s=time.perf_counter() - start, stations=rows)
    if args.kind == "plate":
        report["analytic_np_per_m"] = math.sqrt(math.pi * args.frequency * MU0 / 5.8e7) / (
            math.sqrt(MU0 / (EPS0 * 2.25)) * .010)
    print(json.dumps({key: value for key, value in report.items() if key != "stations"}))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
