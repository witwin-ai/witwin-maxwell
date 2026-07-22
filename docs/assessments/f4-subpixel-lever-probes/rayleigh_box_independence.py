"""Verify sigma_max_ratio box-independence for the Rayleigh RCS re-anchor.

Mirrors the tfsf_rayleigh_rcs_validation fixture body but sweeps box_half so we
can confirm the edge-native far field remains box-size independent (the invariant
that justified the original anchor), just at a new absolute polarizability level.
"""
import math

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess import (
    NearFieldFarFieldTransformer,
    compute_bistatic_rcs,
    equivalent_surface_currents_from_monitor,
)


def sigma_max_ratio(box_half):
    frequency = 1.0e9
    amplitude = 1.0
    sphere_radius = 0.01
    surface_monitor = mw.ClosedSurfaceMonitor.box(
        "huygens",
        position=(0.0, 0.0, 0.0),
        size=(2.0 * box_half, 2.0 * box_half, 2.0 * box_half),
        frequencies=(frequency,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="rayleigh_sphere",
            geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=sphere_radius),
            material=mw.Material(eps_r=4.0),
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization="Ex",
            source_time=mw.CW(frequency=frequency, amplitude=amplitude),
            injection=mw.TFSF(bounds=((-0.04, 0.04), (-0.04, 0.04), (-0.04, 0.04))),
            name="tfsf_pw",
        )
    )
    scene.add_monitor(surface_monitor)

    reference_scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
    )
    reference_scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization="Ex",
            source_time=mw.CW(frequency=frequency, amplitude=amplitude),
            injection=mw.TFSF(bounds=((-0.04, 0.04), (-0.04, 0.04), (-0.04, 0.04))),
            name="tfsf_pw",
        )
    )
    reference_scene.add_monitor(mw.PointMonitor("incident_probe", (0.0, 0.0, 0.0), fields=("Ex",)))

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=12, transient_cycles=30),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    reference_result = mw.Simulation.fdtd(
        reference_scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=12, transient_cycles=30),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    incident_amplitude = abs(reference_result.monitor("incident_probe")["data"].item())
    del reference_result

    currents = equivalent_surface_currents_from_monitor(result, surface_monitor.name)
    transformer = NearFieldFarFieldTransformer(currents, solver=result.solver, device="cuda")
    theta = torch.linspace(0.0, torch.pi, 121, device="cuda", dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * torch.pi, 241, device="cuda", dtype=torch.float64)
    theta_grid, phi_grid = torch.broadcast_tensors(theta[:, None], phi[None, :])
    far_field = transformer.transform(theta_grid, phi_grid, radius=8.0, batch_size=4096)
    rcs = compute_bistatic_rcs(far_field, incident_amplitude=incident_amplitude, c=result.solver.c)

    theta_np = theta_grid.detach().cpu().numpy().astype(float)
    phi_np = phi_grid.detach().cpu().numpy().astype(float)
    sigma = rcs["rcs"].detach().cpu().numpy().astype(float)
    wave_number = 2.0 * math.pi * frequency / result.solver.c
    rayleigh_factor = abs((4.0 - 1.0) / (4.0 + 2.0))
    reference = (
        4.0 * math.pi * (wave_number**4) * (sphere_radius**6) * (rayleigh_factor**2)
        * (1.0 - (np.sin(theta_np) * np.cos(phi_np)) ** 2)
    )
    return float(np.max(sigma)) / float(np.max(reference))


if __name__ == "__main__":
    for bh in (0.05, 0.06, 0.07):
        r = sigma_max_ratio(bh)
        print(f"box_half={bh:.2f}  sigma_max_ratio={r:.4f}")
