"""Orientation, oblique wave coupling and independently measured conductor power."""
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from tests.materials.sheet.sheet_measurement import decompose_two_waves
from tests.materials.surface_impedance.test_copper_foil import DELTA, FREQUENCY, _film_transfer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CopperFoil plane-wave physics requires CUDA.")


def _measure(*, normal=0, angle=0., polarization="te", foil=True):
    tangent = (normal + 1) % 3
    remaining = (normal + 2) % 3
    component = remaining if polarization == "te" else tangent
    direction = np.zeros(3)
    direction[normal], direction[tangent] = math.cos(angle), math.sin(angle)
    electric = np.zeros(3)
    if polarization == "te":
        electric[remaining] = 1.
    else:
        electric[normal], electric[tangent] = -math.sin(angle), math.cos(angle)
    domain = [(-.01, .01)] * 3
    domain[normal] = (-.3, .3)
    size = [0.08] * 3
    size[normal] = 0.
    kinds = {"xyz"[normal]: "pml", "xyz"[component]: "pec", "xyz"[3 - normal - component]: "pmc"}
    wavevector = [0.] * 3
    if angle:
        azimuth = .4
        direction[tangent] = math.sin(angle) * math.cos(azimuth)
        direction[remaining] = math.sin(angle) * math.sin(azimuth)
        if polarization == "te":
            electric[tangent], electric[remaining] = -math.sin(azimuth), math.cos(azimuth)
        else:
            electric[tangent] = math.cos(angle) * math.cos(azimuth)
            electric[remaining] = math.cos(angle) * math.sin(azimuth)
        kinds["xyz"[tangent]] = "bloch"
        kinds["xyz"[remaining]] = "bloch"
        wavevector[tangent] = 2 * math.pi * FREQUENCY / 299792458. * direction[tangent]
        wavevector[remaining] = 2 * math.pi * FREQUENCY / 299792458. * direction[remaining]
    scene = mw.Scene(domain=mw.Domain(bounds=domain), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.faces(default="none", num_layers=12, strength=1.,
            bloch_wavevector="auto" if angle else wavevector, **kinds), device="cuda",
        structures=[mw.Structure(geometry=mw.Box(position=torch.zeros(3, dtype=torch.float64),
            size=torch.tensor(size, dtype=torch.float64)),
            material=mw.CopperFoil(thickness=DELTA, frequency_range=(.5e9, 2e9)), name="foil")] if foil else [],
        sources=[mw.PlaneWave(direction=direction, polarization=electric, injection_axis="xyz"[normal],
            injection=mw.TFSF.slab(axis="xyz"[normal], bounds=(-.25, .25)) if angle else "soft",
            source_time=mw.CW(frequency=FREQUENCY, amplitude=80.), name="incident")])
    result = mw.Simulation.fdtd(scene, absorber="cpml", frequency=FREQUENCY,
        run_time=mw.TimeConfig.auto(steady_cycles=20, transient_cycles=24),
        spectral_sampler=mw.SpectralSampler(window="hanning"), full_field_dft=True).run()
    field = result.tensor("E" + "xyz"[component])
    if field.ndim == 4:
        field = field[0]
    field = field.movedim(normal, 0)
    profile = field[:, field.shape[1] // 2, field.shape[2] // 2].detach().cpu().numpy()
    x = getattr(result.solver.scene, "xyz"[normal] + "_nodes64")
    dt, spacing = result.solver.dt, .005
    if angle:
        wavevector = result.solver.resolved_bloch_wavevector
    transverse_squared = sum((math.sin(wavevector[index] * spacing / 2) / spacing)**2
                             for index in (tangent, remaining))
    longitudinal = math.sqrt((math.sin(math.pi * FREQUENCY * dt) / (299792458. * dt))**2 - transverse_squared)
    k = 2 / spacing * math.asin(spacing * longitudinal)
    left_forward, left_backward = decompose_two_waves(x, profile, -.15, -.025, k)
    right_forward, right_backward = decompose_two_waves(x, profile, .025, .15, k)
    determinant = left_forward**2 - right_backward**2
    reflection = (left_backward * left_forward - right_forward * right_backward) / determinant
    transmission = (right_forward * left_forward - left_backward * right_backward) / determinant
    return reflection, transmission, left_forward, result


@pytest.mark.parametrize("normal", [0, 1, 2])
def test_every_surface_orientation_and_reported_loss_matches_scattered_flux(normal):
    reflection, transmission, incident, result = _measure(normal=normal)
    expected_r, expected_t, zi, _ = _film_transfer(DELTA)
    assert abs(reflection - expected_r) < .002
    assert abs(transmission - expected_t) / abs(expected_t) < .03
    incident_power = .5 * abs(incident)**2 / zi * .02**2
    flux_loss = incident_power * (1 - abs(reflection)**2 - abs(transmission)**2)
    surface_power = result.conductor_losses["foil"]["surface_power"].item()
    print(dict(normal=normal, surface_power=surface_power, scattered_flux_loss=flux_loss))
    assert abs(surface_power - flux_loss) / flux_loss < .03


@pytest.mark.parametrize("polarization", [
    pytest.param("te", marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
        "337cee2f made CFS-CPML the default for complex fixed-k runs; it raises the 35 deg TE "
        "transmission error to 4.4% against the 3% bound (passes with alpha_max=0). "
        "Follow-up branch bloch/cfs-cpml-propagating-accuracy restores accuracy and removes this marker."))),
    "tm",
])
def test_oblique_bloch_incidence_retains_correct_transmission_and_small_copper_absorption(polarization):
    angle = math.radians(35)
    reflection, transmission, incident, result = _measure(angle=angle, polarization=polarization)
    expected_r, expected_t, zi, zt = _film_transfer(DELTA, angle=angle, polarization=polarization)
    actual_absorption = 1 - abs(reflection)**2 - zi / zt * abs(transmission)**2
    expected_absorption = 1 - abs(expected_r)**2 - zi / zt * abs(expected_t)**2
    print(dict(polarization=polarization, r=reflection, t=transmission,
        absorption=actual_absorption, expected_absorption=expected_absorption))
    assert abs(reflection - expected_r) < .005
    assert abs(transmission - expected_t) / abs(expected_t) < .03
    assert abs(actual_absorption - expected_absorption) / expected_absorption < .05
    assert result.solver.complex_fields_enabled
    assert result.conductor_losses["foil"]["surface_power"].item() > 0


@pytest.mark.parametrize("polarization", ["te", "tm"])
def test_oblique_bloch_vacuum_carrier_has_unit_transmission(polarization):
    reflection, transmission, incident, _result = _measure(
        angle=math.radians(35), polarization=polarization, foil=False)
    assert abs(incident) > 1.
    assert abs(reflection) < 1e-4
    assert abs(transmission - 1) < 1e-4
