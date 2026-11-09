"""Shared physical fixture and analytic oracle for fixed-k_parallel acceptance."""

import sysconfig
import os

os.environ.setdefault("CUDA_HOME", os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13"))

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0
from witwin.maxwell.fdtd.runtime.stepping import _electric_field_energy
from witwin.maxwell.postprocess.floquet import reflection_transmission

F0 = 1.e9
LAMBDA0 = C_0 / F0
K0 = 2 * math.pi / LAMBDA0
THICKNESS = .37 * LAMBDA0
SLAB_CENTER = .013 * LAMBDA0
FRONT = SLAB_CENTER - THICKNESS / 2
BACK = SLAB_CENTER + THICKNESS / 2
REFLECTION_Z = -.4 * LAMBDA0
TRANSMISSION_Z = .4 * LAMBDA0
PERIOD = (.1 * LAMBDA0, .1 * LAMBDA0)
FREQUENCIES = tuple(F0 * v for v in (.8, .85, .9, .95, 1., 1.05, 1.1, 1.15, 1.2))
LIGHT_LINE_MARGIN = .05
SPECTRAL_TAIL = 1.e-8


def make_pulse(kpar, f_min, f_max):
    """Gaussian with spectral amplitude <=2e-8 at 1.05 times the light line.

    f_min/f_max specify the label band in Hz, above the guarded cutoff. The
    carrier is the band midpoint; width is bounded by both the label band
    and cutoff separation. Each of the real pulse's two Gaussian spectral
    lobes is <=1e-8 there. Eight-sigma turn-on suppresses truncation leakage.
    This is Gaussian spectral suppression, not compact spectral support.
    """
    frequency = (f_min + f_max) / 2
    cutoff = C_0 * abs(kpar) / (2 * math.pi)
    width = min((f_max - f_min) / 2,
                (frequency - cutoff * (1 + LIGHT_LINE_MARGIN)) / math.sqrt(2 * math.log(1 / SPECTRAL_TAIL)))
    return mw.GaussianPulse(frequency=frequency, fwidth=width, delay=8 / (2 * math.pi * width))


def oblique_airy(frequencies, kpar, eps_r, polarization, thickness=THICKNESS):
    """Vacuum/slab/vacuum electric r/t at the two physical slab faces.

    exp(-i omega t) convention; TM has the same positive tangential electric
    direction for incident and reflected waves. Inputs are propagating in
    vacuum. Equal exterior media make abs(r)^2+abs(t)^2 the power balance.
    """
    frequencies = torch.as_tensor(frequencies, dtype=torch.float64)
    k = 2 * math.pi * frequencies / C_0
    kz0 = torch.sqrt(k.square() - kpar ** 2)
    kz1 = torch.sqrt(eps_r * k.square() - kpar ** 2)
    if polarization == "TE":
        y0, y1 = kz0, kz1
    else:
        y0, y1 = 1 / kz0, eps_r / kz1
    rho = (y0 - y1) / (y0 + y1)
    phase = torch.exp(1j * kz1 * thickness)
    denominator = 1 - rho.square() * phase.square()
    return rho * (1 - phase.square()) / denominator, (1 - rho.square()) * phase / denominator


def make_scene(resolution, kpar, polarization, eps_r, source_time, *, auto=False):
    sine = kpar * C_0 / (2 * math.pi * source_time.frequency)
    cosine = math.sqrt(1 - sine ** 2)
    # Explicit nodes keep the physical planes fixed throughout refinement.
    x = np.linspace(-PERIOD[0] / 2, PERIOD[0] / 2, resolution // 10 + 1)
    y = np.linspace(-PERIOD[1] / 2, PERIOD[1] / 2, resolution // 10 + 1)
    z = np.linspace(-LAMBDA0, LAMBDA0, 2 * resolution + 1)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((x[0], x[-1]), (y[0], y[-1]), (z[0], z[-1]))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.faces(
            x="bloch", y="bloch", z="pml", num_layers=resolution // 2,
            bloch_wavevector="auto" if auto else (kpar, 0., 0.),
        ),
        subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 128), averaging="polarized"),
        device="cuda",
    )
    if eps_r != 1:
        scene.add_structure(mw.Structure(
            geometry=mw.Box(position=(0., 0., SLAB_CENTER), size=(LAMBDA0, LAMBDA0, THICKNESS)),
            material=mw.Material(eps_r=eps_r), name="slab",
        ))
    scene.add_source(mw.PlaneWave(
        direction=(sine, 0., cosine),
        polarization=(0., 1., 0.) if polarization == "TE" else (cosine, 0., -sine),
        source_time=source_time, injection=mw.TFSF.slab(axis="z", bounds=(-.65 * LAMBDA0, .65 * LAMBDA0)),
        name="incident",
    ))
    for name, position in (("reflection", REFLECTION_Z), ("transmission", TRANSMISSION_Z)):
        scene.add_monitor(mw.PlaneMonitor(name, axis="z", position=position, fields=("Ex", "Ey")))
    return scene


def make_simulation(resolution, kpar, polarization, eps_r, frequencies=FREQUENCIES, *,
                    source_time=None, cw=False, cw_transient_cycles=300, cw_auto=False,
                    full_field_dft=False, cuda_graph=True, time_steps=None, shutoff=None):
    if source_time is None:
        source_time = (mw.CW(frequency=frequencies[0], amplitude=1.) if cw else
                       make_pulse(kpar, min(frequencies), max(frequencies)))
    scene = make_scene(resolution, kpar, polarization, eps_r, source_time, auto=cw and cw_auto)
    scene.add_monitor(mw.PlaneMonitor("scattered", axis="z", position=-.8 * LAMBDA0, fields=("Ex", "Ey")))
    return mw.Simulation.fdtd(
        scene, frequencies=frequencies,
        run_time=(mw.TimeConfig(time_steps=time_steps) if time_steps is not None else
                  mw.TimeConfig.auto(transient_cycles=cw_transient_cycles if cw else 0,
                                     steady_cycles=100 if cw else 80)),
        spectral_sampler=mw.SpectralSampler(window="hanning" if cw else "none"),
        full_field_dft=full_field_dft, cuda_graph=cuda_graph,
        absorber="cpml", shutoff=(0. if cw else 1.e-10) if shutoff is None else shutoff,
    )


def run_pair(resolution, kpar, polarization, eps_r, frequencies=FREQUENCIES, *, cw=False,
             cw_transient_cycles=300, cw_auto=True, source_time=None, shutoff=None, time_steps=None,
             reference_bins=None):
    results = []
    for material in (1., eps_r):
        simulation = make_simulation(resolution, kpar, polarization, material, frequencies,
                                     cw=cw, cw_transient_cycles=cw_transient_cycles, cw_auto=cw_auto,
                                     source_time=source_time, shutoff=shutoff, time_steps=time_steps)
        prepared = simulation.prepare()
        result = prepared.run()
        solver = prepared.solver
        resolved_kpar = float(solver.resolved_bloch_wavevector[0])
        energy = _electric_field_energy(solver)
        assert torch.isfinite(energy), "Nonfinite final electric energy"
        tail_ratio = float(energy / solver._shutoff_peak) if not cw else None
        print(dict(resolution=resolution, kpar=resolved_kpar, polarization=polarization, eps_r=material,
                   cw=cw, transient_cycles=cw_transient_cycles if cw else 0,
                   steps=result.stats()["steps_run"], tail_energy_ratio=tail_ratio), flush=True)
        if not cw and shutoff is None:
            assert result.stats()["shutoff_triggered"], f"Pulse missed 1e-10 energy shutoff: tail/peak={tail_ratio}"
        results.append(result)
    kwargs = dict(reflection_monitor_name="reflection", transmission_monitor_name="transmission",
                  period=PERIOD, bloch_wavevector=(resolved_kpar, 0.), frequencies=frequencies,
                  background=mw.Material(eps_r=1.), incident_polarization=polarization,
                  light_line_margin=LIGHT_LINE_MARGIN)
    data = reflection_transmission(results[1], results[0], **kwargs)
    zero = (data["orders"] == 0).all(-1).nonzero().flatten()[0]
    pol = data["polarizations"].index(polarization)
    if not cw:
        line_data = reflection_transmission(results[1], **kwargs)
        valid = ~data["mask"][:, zero]
        if reference_bins is not None:
            # Compare only bins the source spectrum drives strongly: a normalized
            # coefficient in a weak bin amplifies residual tail and truncation error.
            valid = valid & torch.as_tensor(reference_bins, device=valid.device)
        for key in ("r", "t"):
            gap = abs(line_data[key][valid, zero, pol] - data[key][valid, zero, pol])
            print(dict(incident_reference=key, maximum_absolute_gap=float(gap.max())), flush=True)
            assert (gap < 1.e-4).all(), "Line and matched-run incident references disagree"
    r = data["r"][:, zero, pol].detach().cpu().to(torch.complex128)
    t = data["t"][:, zero, pol].detach().cpu().to(torch.complex128)
    mask = data["mask"][:, zero].detach().cpu()
    k = 2 * math.pi * torch.tensor(frequencies, dtype=torch.float64) / C_0
    kz = torch.sqrt(k.square() - resolved_kpar ** 2)
    # Move the electric r/t reference planes to the physical slab faces.
    r = r * torch.exp(-2j * kz * (FRONT - REFLECTION_Z))
    t = t * torch.exp(-1j * kz * (TRANSMISSION_Z - REFLECTION_Z - THICKNESS))
    del data, results, result, prepared, simulation, solver
    torch.cuda.empty_cache()
    return dict(r=r, t=t, kpar=resolved_kpar, mask=mask)
