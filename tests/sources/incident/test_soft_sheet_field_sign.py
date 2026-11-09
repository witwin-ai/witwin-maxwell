"""Absolute E/H polarity of power-normalized open launch sheets.

The analytic reference includes Yee dispersion. Observers carry physical Yee
time labels, so no clock offset remains. No measured phase or amplitude is fitted away.
"""

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.spatial import soft_plane_wave_coordinate
from witwin.maxwell.scene import prepare_scene


C0 = 299792458.0
ETA0 = 376.730313668
FREQUENCY = 2.0e9
OMEGA = 2.0 * math.pi * FREQUENCY
DX = 0.025


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _weighted_ratio(actual, reference):
    # vdot implements the |reference|^2-weighted ratio without dividing by
    # individual samples. A common E/H reversal must give R near -1.
    return np.vdot(reference, actual) / np.vdot(reference, reference).real


def _assert_positive_field(actual, analytic, clock, record_property, name, tolerance):
    ratio = _weighted_ratio(actual, analytic)
    record_property(name, str(ratio))
    assert ratio.real > 0.5, (name, ratio, clock)
    assert abs(ratio - clock) < tolerance, (name, ratio, clock)


def _run(scene, *, pulse):
    simulation = mw.Simulation.fdtd(
        scene, absorber="cpml", frequencies=[FREQUENCY], backend="cpu_reference", cuda_graph=False,
        full_field_dft=False, run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(
            window="none" if pulse else "hanning", normalize_source=True,
        ),
    )
    dt = simulation.prepare().solver.dt
    simulation.config.run_time = (mw.TimeConfig(time_steps=math.ceil(10e-9 / dt)) if pulse
                                  else mw.TimeConfig.auto(steady_cycles=12, transient_cycles=15))
    prepared = simulation.prepare()
    result = prepared.run()
    # The one-axis Yee dispersion relation, independently of the compiler's
    # numerical-wavenumber helper. Vacuum numerical impedance is ETA0.
    h = float(getattr(prepared.solver.scene, "d" + scene.sources[0].injection_axis + "_primal64").min())
    k = 2.0 * math.asin(math.sin(OMEGA * dt / 2.0) * h / (C0 * dt)) / h
    return result, prepared.solver.scene, k, dt, h


@pytest.mark.parametrize("pulse", (False, True), ids=("cw", "pulse"))
@pytest.mark.parametrize("axis,e_axis", ((0, 1), (0, 2), (2, 0), (2, 1)),
                         ids=("x_Ey", "x_Ez", "z_Ex", "z_Ey"))
@pytest.mark.parametrize("sign", (1, -1), ids=("forward", "reverse"))
def test_soft_plane_wave_absolute_e_and_h_sign(axis, e_axis, sign, pulse, record_property):
    direction = np.eye(3)[axis] * sign
    polarization = np.eye(3)[e_axis]
    magnetic = np.cross(direction, polarization)
    h_axis = int(np.argmax(abs(magnetic)))
    axis_name = "xyz"[axis]
    e_name, h_name = "E" + "xyz"[e_axis], "H" + "xyz"[h_axis]
    bounds = [(-2.0 * DX, 2.0 * DX)] * 3
    bounds[axis] = (-0.64, 0.64)
    # PEC normal to E and PMC normal to H admit an exactly uniform transverse
    # vacuum wave, with no aperture diffraction or unsupported periodic kernel.
    boundary = mw.BoundarySpec.pml(num_layers=8).with_faces(**{
        "xyz"[e_axis] + "_low": "pec",
        "xyz"[e_axis] + "_high": "pec",
        "xyz"[h_axis] + "_low": "pmc",
        "xyz"[h_axis] + "_high": "pmc",
    })
    source_time = (mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9) if pulse
                   else mw.CW(frequency=FREQUENCY))
    scene = mw.Scene(domain=mw.Domain(bounds=tuple(bounds)), grid=mw.GridSpec.uniform(DX),
                     boundary=boundary, device="cpu")
    scene.add_source(mw.PlaneWave(direction=direction, polarization=polarization,
                                 injection_axis=axis_name, source_time=source_time))
    scene.add_monitor(mw.PlaneMonitor("cut", axis="xyz"[h_axis], position=0.0,
                                     fields=(e_name, h_name), frequencies=[FREQUENCY]))
    result, grid, k, dt, h = _run(scene, pulse=pulse)
    launch = soft_plane_wave_coordinate(grid, axis_name, sign)
    ranges = grid.domain_range
    area = math.prod(ranges[2 * a + 1] - ranges[2 * a] for a in range(3) if a != axis)
    e0 = math.sqrt(2.0 * ETA0 / (area * math.cos(k * h / 2.0)))
    # Physical Yee observer labels: source-normalized E and synchronized H match the
    # analytic incident reference at the same physical time (observer time origin).
    clock = 1.0 + 0.0j
    components = result.monitor("cut")["components"]
    ratios = {}
    tangential = [a for a in range(3) if a != h_axis]
    for name, amplitude in ((e_name, e0), (h_name, e0 * magnetic[h_axis] / ETA0)):
        component = components[name]
        coords = np.meshgrid(*[np.asarray(c) for c in component["coords"]], indexing="ij")
        distance = sign * (coords[tangential.index(axis)] - launch)
        mask = (distance > 0.2) & (distance < 0.65)
        analytic = amplitude * np.exp(1j * k * distance[mask])
        actual = component["data"].detach().numpy()[mask]
        # Deterministic CPU reference: the largest measured |ratio - 1| over all 16 cases is 1.6e-5.
        _assert_positive_field(actual, analytic, clock, record_property, name, tolerance=1.0e-4)
        ratios[name] = _weighted_ratio(actual, analytic)
    # In addition to H's absolute complex ratio, its relative sign must give
    # positive Poynting flux along the prescribed direction.
    assert (ratios[e_name] * ratios[h_name].conjugate()).real > 0.5


def test_soft_gaussian_beam_on_axis_absolute_sign(record_property):
    waist = 0.20
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(DX), boundary=mw.BoundarySpec.pml(num_layers=8), device="cpu",
    )
    launch = soft_plane_wave_coordinate(prepare_scene(scene), "z", 1)
    scene.add_source(mw.GaussianBeam(
        direction=(0, 0, 1), polarization="Ex", beam_waist=waist, focus=(0, 0, launch),
        injection_axis="z", source_time=mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9),
    ))
    scene.add_monitor(mw.PlaneMonitor("cut", axis="y", position=0.0,
                                     fields=("Ex",), frequencies=[FREQUENCY]))
    result, _, k, dt, h = _run(scene, pulse=True)
    component = result.monitor("cut")["components"]["Ex"]
    x, z = np.meshgrid(*[np.asarray(c) for c in component["coords"]], indexing="ij")
    distance = z - launch
    mask = (abs(x) <= DX) & (distance > 0.15) & (distance < 0.35)
    # Independent paraxial beam propagated from its waist at the launch plane.
    # Use the discrete carrier and its corresponding Rayleigh range, as does
    # the prescribed launch profile; retain curvature and Gouy phase.
    rayleigh = k * waist**2 / 2.0
    radius = waist * np.sqrt(1.0 + (distance[mask] / rayleigh)**2)
    e0 = math.sqrt(4.0 * ETA0 / (math.pi * waist**2 * math.cos(k * h / 2.0)))
    analytic = (e0 * waist / radius * np.exp(-x[mask]**2 / radius**2)
                * np.exp(1j * (k * distance[mask]
                              + k * x[mask]**2 * distance[mask] / (2.0 * (distance[mask]**2 + rayleigh**2))
                              - np.arctan(distance[mask] / rayleigh))))
    _assert_positive_field(component["data"].detach().numpy()[mask], analytic,
                           np.exp(-1j * OMEGA * dt), record_property, "beam_Ex", tolerance=0.10)
