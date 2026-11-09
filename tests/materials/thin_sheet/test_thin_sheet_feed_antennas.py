"""D6 antenna gates with fixed physical geometry and prescribed grid sweeps."""
import gc
import json
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.postprocess import (NearFieldFarFieldTransformer, compute_directivity,
                                       equivalent_surface_currents_from_monitor)

from .harness import rectangle_mesh

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Thin-sheet antenna gates require CUDA")

WAVELENGTH = .3
FREQUENCY = 299792458.0 / WAVELENGTH
FREQUENCIES = FREQUENCY * np.linspace(.6, 1.4, 161)
LENGTH, WIDTH, GAP = .15, .015, .015
NORMALS = ((0, .025, 1), (0, .5, math.sqrt(3)/2))


def cylindrical_strip():
    # Five facets on a cylinder about y; the top facet is 37.5 mm wide.
    # Adjacent full facets turn by 30 degrees; the two end facets are halved.
    first = np.array((.01875, 0.))
    second = first + .0375*np.array((math.cos(math.pi/6), -math.sin(math.pi/6)))
    last = second + .01875*np.array((.5, -math.sqrt(3)/2))
    right = [first, second, last]
    points = [(-point[0], point[1]) for point in reversed(right)] + right
    vertices = [(x, y, z) for y in (-WIDTH/2, WIDTH/2) for x, z in points]
    count = len(points)
    triangles = [triangle for i in range(count-1)
                 for triangle in ((i, i+1, count+i), (i+1, count+i+1, count+i))]
    return mw.SurfaceMesh(vertices, triangles, sheet_model="conformal_two_sided")


def antenna_scene(resolution, *, normal=None, curved=False):
    cell = WAVELENGTH / resolution
    # The native reference requires both gap terminals on x nodes.
    x_shift = 0. if curved else (round(GAP/cell) % 2) * cell/2
    y_shift = 0. if curved else (round(WIDTH/cell) % 2) * cell/2
    terminal_y = -y_shift if normal is None and not curved else 0.
    foil = mw.CopperFoil(thickness=35e-6, frequency_range=(.5e9, 2e9))
    port = mw.LumpedPort(
        "feed", negative=(-GAP/2, terminal_y, 0), positive=(GAP/2, terminal_y, 0),
        voltage_path=mw.AxisPath("x"),
        current_surface=Box(position=(cell/2-x_shift, -y_shift, 0), size=(0, 5*cell, 5*cell)),
        voltage_aperture=Box(position=(0, 0, 0), size=(GAP, WIDTH, 0)))
    if curved:
        structures = [mw.Structure(geometry=cylindrical_strip(), material=foil)]
    elif normal is None:
        structures = [mw.Structure(
            geometry=Box(position=(sign*(LENGTH+GAP)/4, 0, 0),
                         size=((LENGTH-GAP)/2, WIDTH, 0)), material=foil) for sign in (-1, 1)]
    else:
        structures = [mw.Structure(geometry=rectangle_mesh(
            (0, 0, 0), normal, LENGTH/2, WIDTH/2, u_direction=(1, 0, 0),
            sheet_model="conformal_two_sided"), material=foil)]
    return mw.Scene(
        domain=mw.Domain(bounds=((- .24+x_shift, .24+x_shift),
                                 (- .24+y_shift, .24+y_shift), (- .24, .24))),
        grid=mw.GridSpec.uniform(cell),
        boundary=mw.BoundarySpec.pml(num_layers=10), device="cuda", structures=structures,
        ports=[port], monitors=[mw.ClosedSurfaceMonitor.box(
            "radiation", size=(.24, .24, .24), frequencies=[FREQUENCY])])


def measure_antenna(resolution, *, normal=None, curved=False):
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    cell = WAVELENGTH / resolution
    # Keep dt/h fixed below the sheet CFL bound and use the same sampling clock.
    dt = 2e-12 * cell/.005
    result = mw.Simulation.fdtd(
        antenna_scene(resolution, normal=normal, curved=curved), frequencies=FREQUENCIES.tolist(),
        run_time=mw.TimeConfig(time_steps=math.ceil(12e-9/dt), dt=dt),
        cuda_graph=True, full_field_dft=False,
        spectral_sampler=mw.SpectralSampler(window="none"),
        excitations=mw.PortExcitation("feed", source_time=mw.GaussianPulse(
            FREQUENCY, fwidth=.6*FREQUENCY))).run()
    impedance = result.port("feed").z_in.detach().cpu().numpy()
    # First reactance zero is the half-wave resonance, before the higher mode.
    crossing = np.flatnonzero(impedance[:-1].imag * impedance[1:].imag < 0)[0]
    resonance = float(FREQUENCIES[crossing] - impedance[crossing].imag *
                      (FREQUENCIES[crossing+1]-FREQUENCIES[crossing]) /
                      (impedance[crossing+1].imag-impedance[crossing].imag))
    currents = equivalent_surface_currents_from_monitor(result, "radiation", frequency=FREQUENCY)
    far_field = NearFieldFarFieldTransformer(currents, solver=result.solver).transform(
        torch.linspace(0, math.pi, 91, device="cuda", dtype=torch.float64)[:, None],
        torch.linspace(0, 2*math.pi, 73, device="cuda", dtype=torch.float64)[None, :], radius=10.)
    antenna = compute_directivity(far_field)
    data = dict(resolution=resolution, normal=normal, curved=curved, dt=dt,
                impedance_real=impedance.real.tolist(), impedance_imag=impedance.imag.tolist(),
                resonance_hz=resonance, directivity_dbi=float(antenna["directivity_db"].max()),
                peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    print(json.dumps({key: value for key, value in data.items() if not key.startswith("impedance_")}), flush=True)
    del result, antenna, currents, far_field
    gc.collect()
    torch.cuda.empty_cache()
    return data


def rotation_errors(reference, tilted):
    z0 = np.array(reference["impedance_real"]) + 1j*np.array(reference["impedance_imag"])
    z1 = np.array(tilted["impedance_real"]) + 1j*np.array(tilted["impedance_imag"])
    return dict(impedance=float(np.max(np.abs((z1-z0)/z0))),
                directivity_db=abs(tilted["directivity_dbi"]-reference["directivity_dbi"]),
                resonance=abs(tilted["resonance_hz"]/reference["resonance_hz"]-1))


@pytest.fixture(scope="module")
def aligned_reference():
    return {resolution: measure_antenna(resolution) for resolution in (40, 60)}


@pytest.mark.xfail(strict=True, reason="Conformal rim staircase (partial rim chords rejected): converging, ~2% resonance shift at lambda/40, Z_in gate not met; a sub-cell rim treatment would be needed (docs/assessments/conformal-sheet-rim-rejection-phase1a-2026-10-31.md).")
@pytest.mark.parametrize("normal", NORMALS)
def test_f2_rotation_invariance(normal, aligned_reference):
    errors = [rotation_errors(aligned_reference[resolution], measure_antenna(resolution, normal=normal))
              for resolution in (40, 60)]
    print(dict(F2=normal, errors_by_grid=dict(zip((40, 60), errors))), flush=True)
    assert errors[0]["impedance"] <= .03
    assert errors[1]["impedance"] < errors[0]["impedance"]
    assert max(error["directivity_db"] for error in errors) <= .1
    assert max(error["resonance"] for error in errors) <= .005


def convergence_impedances(data, resonance_hz):
    impedance = np.array(data["impedance_real"]) + 1j*np.array(data["impedance_imag"])
    resonance = (np.interp(resonance_hz, FREQUENCIES, impedance.real)
                 + 1j*np.interp(resonance_hz, FREQUENCIES, impedance.imag))
    return np.array((impedance[0], resonance, impedance[-1]))


@pytest.mark.xfail(strict=True, reason="Conformal rim staircase (partial rim chords rejected): converging, ~2% resonance shift at lambda/40, Z_in gate not met; a sub-cell rim treatment would be needed (docs/assessments/conformal-sheet-rim-rejection-phase1a-2026-10-31.md).")
def test_f2b_curved_self_convergence():
    data = [measure_antenna(resolution, curved=True) for resolution in (40, 60, 80)]
    # Use one physical frequency: the finest grid's first reactance zero.
    z40, z60, z80 = [convergence_impedances(case, data[-1]["resonance_hz"]) for case in data]
    changes = np.abs(z60-z40), np.abs(z80-z60)
    directivity = abs(data[2]["directivity_dbi"]-data[1]["directivity_dbi"])
    print(dict(F2b_impedance_changes=[change.tolist() for change in changes],
               F2b_directivity_change_db=directivity), flush=True)
    assert np.all(changes[1] < changes[0])
    assert directivity <= .1


def measure_power(kind, monkeypatch):
    """Exact field-side feed power and independent copper/flux spectra at lambda/60."""
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    scene = antenna_scene(60, normal=NORMALS[1], curved=kind == "curved")
    if kind == "patch":
        scene.structures = [mw.Structure(geometry=rectangle_mesh(
            (0, 0, 0), NORMALS[1], .045, .03, u_direction=(1, 0, 0),
            sheet_model="conformal_two_sided"), material=scene.structures[0].material)]
    frequencies = FREQUENCY*np.array((.8, 1., 1.2))
    scene.monitors = [mw.ClosedSurfaceMonitor.box(
        "radiation", size=(.24, .24, .24), frequencies=frequencies.tolist())]
    prepared = mw.Simulation.fdtd(
        scene, frequencies=frequencies.tolist(), run_time=mw.TimeConfig(time_steps=6000, dt=2e-12),
        cuda_graph=False, full_field_dft=False, spectral_sampler=mw.SpectralSampler(window="none"),
        excitations=mw.PortExcitation("feed", source_time=mw.GaussianPulse(
            FREQUENCY, fwidth=.6*FREQUENCY))).prepare()
    solver = prepared.solver
    sheet, = solver._thin_sheets
    cu = sheet.copper
    e_sum = torch.zeros((3, 2, len(cu["area"])), dtype=torch.complex128, device="cuda")
    z_sum = torch.zeros((3, *cu["state"].shape), dtype=torch.complex128, device="cuda")
    electric = sheet.electric

    def collect(solver):
        previous_state = cu["state"].clone()
        electric(solver)
        pair = cu["pair_position"]
        midpoint_e = torch.cat(((sheet.chord_e+sheet.chord_e_prev)/2,
                               (sheet.e_old[pair]+sheet.e_new[pair])/2)).T
        port, = solver._port_runtimes
        kernel = port.current_phase_weights[port.sample_index, :, None, None]
        e_sum.add_(kernel*midpoint_e[None])
        z_sum.add_(kernel*((previous_state+cu["state"])/2)[None])

    monkeypatch.setattr(sheet, "electric", collect)
    result = prepared.run()
    port, = solver._port_runtimes
    # Pulses use a separate reference period for each output frequency bin.
    scale = solver._spectral_integration_dt*port.frequencies[:, None, None]
    e_hat, z_hat = e_sum*scale, z_sum*scale
    loss = (cu["rates"][None, :, None]*z_hat.abs().square()).sum(1)
    loss += (e_hat.conj()*torch.einsum("ij,fjn->fin", cu["D"].to(e_hat.dtype), e_hat)).real.sum(1)
    copper = .5*(loss*cu["area"]).sum(1)
    radiation = torch.stack([torch.as_tensor(result.monitor("radiation", frequency=f)["flux"])
                             for f in frequencies]).reshape(-1)
    diagnostics = result.port("feed").metadata["sheet_gap"]
    accepted = diagnostics["field_accepted_power"]
    closure = (accepted-radiation-copper).abs()/accepted.abs()
    data = dict(kind=kind, resolution=60, frequencies_hz=frequencies.tolist(),
                accepted=accepted.tolist(), radiation=radiation.tolist(), copper=copper.tolist(),
                load=[0., 0., 0.], port_accepted=diagnostics["port_accepted_power"].tolist(),
                variance=diagnostics["variance_power"].tolist(), closure=closure.tolist(),
                peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    print(json.dumps(dict(F3=data)), flush=True)
    return data


@pytest.mark.parametrize("kind", ["dipole", "patch", "curved"])
def test_f3_closed_box_power_balance(kind, monkeypatch):
    data = measure_power(kind, monkeypatch)
    assert max(data["closure"]) <= .01
