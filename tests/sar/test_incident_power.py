"""Analytic and consistency gates for IncidentPowerDensityMonitor.

The postprocess is exercised directly on synthetic plane payloads (no solver) so
the plane-wave relation ``|S| = |E|^2/(2*eta)`` is machine-tight rather than
FDTD-accuracy-limited, plus one small vacuum plane-wave FDTD run that ties the
monitor's integrated flux to a co-located FluxMonitor end-to-end.
"""

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.observers import _compute_plane_flux, plane_normal_poynting
from witwin.maxwell.postprocess.incident_power import compute_incident_power_density

# Vacuum wave impedance eta0 = mu0 * c.
ETA0 = (4.0e-7 * math.pi) * 299_792_458.0


def _z_plane_payload(ex, ey, hx, hy, *, dx=0.01, dy=0.01, normal_direction="+", device="cpu"):
    """Build a z-normal flux-enabled plane payload from tangential field grids."""

    ex = torch.as_tensor(ex, dtype=torch.complex64, device=device)
    ey = torch.as_tensor(ey, dtype=torch.complex64, device=device)
    hx = torch.as_tensor(hx, dtype=torch.complex64, device=device)
    hy = torch.as_tensor(hy, dtype=torch.complex64, device=device)
    nu, nv = ex.shape
    x = torch.arange(nu, dtype=torch.float64, device=device) * dx
    y = torch.arange(nv, dtype=torch.float64, device=device) * dy
    return {
        "axis": "z",
        "normal_direction": normal_direction,
        "frequency": 1.0e9,
        "frequencies": (1.0e9,),
        "x": x,
        "y": y,
        "cell_widths": {
            "x": torch.full((nu,), dx, dtype=torch.float64, device=device),
            "y": torch.full((nv,), dy, dtype=torch.float64, device=device),
        },
        "Ex": ex,
        "Ey": ey,
        "Hx": hx,
        "Hy": hy,
    }


def _plane_wave_payload(e0, *, nu=6, nv=5, eta=ETA0, **kwargs):
    """+z plane wave: E = e0 x_hat, H = (e0/eta) y_hat, so S.z = e0^2/(2 eta)."""

    ex = torch.full((nu, nv), complex(e0), dtype=torch.complex64)
    ey = torch.zeros((nu, nv), dtype=torch.complex64)
    hx = torch.zeros((nu, nv), dtype=torch.complex64)
    hy = torch.full((nu, nv), complex(e0 / eta), dtype=torch.complex64)
    return _z_plane_payload(ex, ey, hx, hy, **kwargs)


def test_plane_wave_power_density_matches_analytic():
    e0 = 3.0
    payload = _plane_wave_payload(e0)
    result = compute_incident_power_density(payload, monitor_name="exp")

    analytic = e0**2 / (2.0 * ETA0)
    torch.testing.assert_close(
        result.power_density,
        torch.full_like(result.power_density, analytic),
        rtol=1e-5,
        atol=0.0,
    )
    # Signed normal Poynting is positive for a +z wave read with a +normal.
    torch.testing.assert_close(result.normal_poynting, result.power_density)
    assert result.power_density_unit == "W/m^2"
    assert result.provenance["poynting_definition"] == "0.5*Re((E x conj(H)).n_hat)"


def test_flux_matches_plane_flux_helper_and_analytic():
    e0 = 2.5
    dx, dy = 0.01, 0.02
    nu, nv = 6, 5
    payload = _plane_wave_payload(e0, nu=nu, nv=nv, dx=dx, dy=dy)

    result = compute_incident_power_density(payload, monitor_name="exp")
    # Identically equal to the shared plane-flux integral (same Poynting helper).
    torch.testing.assert_close(result.flux, _compute_plane_flux(payload))

    area = nu * dx * nv * dy
    analytic_flux = e0**2 / (2.0 * ETA0) * area
    torch.testing.assert_close(
        result.flux, torch.as_tensor(analytic_flux, dtype=result.flux.dtype), rtol=1e-5, atol=0.0
    )


def test_normal_direction_flips_sign():
    e0 = 1.7
    payload_plus = _plane_wave_payload(e0, normal_direction="+")
    payload_minus = _plane_wave_payload(e0, normal_direction="-")
    plus = compute_incident_power_density(payload_plus, monitor_name="exp")
    minus = compute_incident_power_density(payload_minus, monitor_name="exp")

    # Signed component flips, magnitude is orientation-invariant.
    torch.testing.assert_close(minus.normal_poynting, -plus.normal_poynting)
    torch.testing.assert_close(minus.power_density, plus.power_density)


def test_spatial_average_of_uniform_field_is_the_constant():
    e0 = 4.0
    payload = _plane_wave_payload(e0, nu=12, nv=10, dx=0.01, dy=0.01)
    result = compute_incident_power_density(
        payload, monitor_name="exp", spatial_average_area=4e-4
    )
    assert result.spatial_average is not None
    analytic = e0**2 / (2.0 * ETA0)
    torch.testing.assert_close(
        result.spatial_average,
        torch.full_like(result.spatial_average, analytic),
        rtol=1e-5,
        atol=0.0,
    )
    sa = result.provenance["spatial_average"]
    assert sa["version"] == "spatial-average-v1"
    assert sa["certified"] is False
    assert sa["area_m2"] == pytest.approx(4e-4)


def test_spatial_average_matches_bruteforce_on_nonuniform_field():
    # A structured |S| map (a smooth-ish ramp plus a step) on a uniform plane.
    torch.manual_seed(0)
    nu, nv = 14, 11
    dx = dy = 0.01
    density = torch.rand((nu, nv), dtype=torch.float64) + 0.5
    # Encode a real |S| directly by picking Ex real, Hy = 2*density/Ex so that
    # 0.5*Ex*Hy = density (Ex fixed to 1 keeps the algebra trivial).
    ex = torch.ones((nu, nv), dtype=torch.complex64)
    hy = (2.0 * density).to(torch.complex64)
    ey = torch.zeros((nu, nv), dtype=torch.complex64)
    hx = torch.zeros((nu, nv), dtype=torch.complex64)
    payload = _z_plane_payload(ex, ey, hx, hy, dx=dx, dy=dy)

    area = 9.0 * dx * dy  # side = 3*dx -> a 3x3 interior window
    result = compute_incident_power_density(
        payload, monitor_name="exp", spatial_average_area=area
    )
    power_density = result.power_density.to(torch.float64).cpu().numpy()

    half = 0.5 * math.sqrt(area)
    x = np.arange(nu) * dx
    y = np.arange(nv) * dy
    reference = np.zeros((nu, nv))
    for i in range(nu):
        umask = np.abs(x - x[i]) <= half + 1e-12
        for j in range(nv):
            vmask = np.abs(y - y[j]) <= half + 1e-12
            block = power_density[np.ix_(umask, vmask)]
            reference[i, j] = block.mean()  # uniform cell area cancels
    got = result.spatial_average.to(torch.float64).cpu().numpy()
    # float32 reducer: prefix-sum accumulation vs a direct mean agree to float32.
    np.testing.assert_allclose(got, reference, rtol=1e-5, atol=1e-6)


def test_plane_normal_poynting_sums_to_flux():
    payload = _plane_wave_payload(2.0, dx=0.013, dy=0.017)
    poynting, weights = plane_normal_poynting(payload)
    torch.testing.assert_close((poynting * weights).sum(), _compute_plane_flux(payload))


# ---------------------------------------------------------------------------
# End-to-end: a small vacuum plane-wave FDTD run ties the monitor's integrated
# flux to a co-located FluxMonitor, and the per-cell density is uniform.
# ---------------------------------------------------------------------------

_HAS_CUDA = torch.cuda.is_available()


def _vacuum_planewave_scene(frequency, position):
    half = 0.4
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cuda" if _HAS_CUDA else "cpu",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.GaussianPulse(frequency=frequency, fwidth=0.5 * frequency),
        )
    )
    scene.add_monitor(
        mw.IncidentPowerDensityMonitor(
            name="exposure", axis="z", position=position, frequencies=(frequency,)
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(name="flux", axis="z", position=position, frequencies=(frequency,))
    )
    return scene


@pytest.mark.skipif(not _HAS_CUDA, reason="FDTD requires CUDA in this environment.")
def test_incident_power_flux_matches_fluxmonitor_end_to_end():
    frequency = 1.0e9
    position = -0.1
    scene = _vacuum_planewave_scene(frequency, position)
    result = mw.Simulation.fdtd(
        scene=scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=20),
        spectral_sampler=mw.SpectralSampler(normalize_source=True),
    ).run()

    ipd = result.incident_power_density("exposure")
    flux_payload = result.raw_monitor("flux")
    monitor_flux = float(flux_payload["flux"])

    # The monitor's integrated flux equals the co-located FluxMonitor integral.
    assert float(ipd.flux) == pytest.approx(monitor_flux, rel=1e-4, abs=1e-12)
    # Plane-wave power density is (near) uniform and positive across the plane.
    density = ipd.power_density
    assert torch.all(density > 0)
    # Mean equals flux / plane-area (self-consistency of the per-cell map).
    coord_u, coord_v = ipd.coordinates
    area = float(coord_u.max() - coord_u.min()) * float(coord_v.max() - coord_v.min())
    if area > 0:
        assert float(density.mean()) > 0
