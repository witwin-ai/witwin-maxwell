"""CUDA physics gates for the annular coaxial lumped port and resistor.

Two vacuum coax rigs on 0.25 mm cells: a grid-aligned square coax (PEC Boxes) and a
round coax (Cylinder inner radius 1 mm, 96-gon shield 2.3 mm). The staircase PEC rule
(a node is a conductor node when it lies inside or on a PEC body; an edge is shorted
iff both of its endpoints are conductor nodes) makes both an exact discrete TEM line,
so every gate is physical and none fits a line parameter:

* Reference impedance. The TEM mode of a Yee line whose PEC is a node set is
  ``E_t = -grad_h(phi)`` of the transverse lattice potential (1 on the inner, 0 on the
  outer conductor-node set), and ``V / I = eta0 / C~`` holds identically, with
  ``C~ = C' / eps0`` the lattice capacitance (sum of ``(d phi)^2`` over the free
  transverse edges of the compiled solver). For the round line this defines
  capacitance-equivalent radii of the staircased conductors,
  ``(eta0 / 2 pi) ln(b_eff / a_eff) = eta0 / C~``. Equal-area radii are not used: they
  ignore the field concentration at the stair corners (the closed conductor cells give
  67.1 ohm at 0.25 mm, against 56.9 ohm for the line). Round ``Z_ref`` = 56.910 /
  54.236 / 52.068 ohm at 0.25 / 0.125 / 0.0625 mm converges at first order to the
  nominal ``60 ln(b / a)`` = 49.940 ohm; the square line has ``Z_ref`` = 28.448 ohm.
* The port-launched wave travels at c (plane monitors), and two passive probes see a
  uniform line with ``Zc = Z_ref`` and ``beta = k0``.
* The inner-conductor ends at the feed and load planes are lumped shunt capacitances,
  measured once with an open standard (no resistor). With them, ideal line theory
  with ``Z_ref`` and k0 predicts every fed load, the resistor adds exactly ``1 / R``
  and the round two-port sweep matches the cascade.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="coaxial physics gates require CUDA")

ETA0 = 376.730313668
C0 = 299792458.0
H = 0.25e-3
A = 1.0e-3
B = 2.3e-3
Z0, Z1, LENGTH, HALF = 1.0e-3, 16.0e-3, 18.0e-3, 3.0e-3
PROBES = (("probe_a", 4.0e-3), ("probe_b", 13.0e-3))
SQUARE_PLANES = (4.0e-3, 6.0e-3, 8.0e-3, 10.0e-3, 12.0e-3)
# Round line: the conductor-end near fields decay about 3x per 0.5 mm (1e-5 of the TEM
# functional 3 mm from an end), which biases a 1 GHz fit; keep the planes 5 mm inside.
ROUND_PLANES = tuple(1.0e-3 * (6.0 + 0.5 * index) for index in range(11))
FREQUENCIES = (1.0e9, 2.0e9, 3.0e9)
K0 = 2 * math.pi * np.asarray(FREQUENCIES) / C0
PULSE = mw.GaussianPulse(frequency=2.0e9, fwidth=1.5e9)
NOMINAL_IMPEDANCE = ETA0 / (2 * math.pi) * math.log(B / A)


def _coax(name, z, **kwargs):
    return mw.CoaxialLumpedPort(name, center=(0.0, 0.0, z), inner_diameter=2 * A, outer_diameter=2 * B,
                                normal_axis="z", **kwargs)


def _add_plane_monitors(scene, planes):
    for index, z in enumerate(planes):
        scene.add_monitor(mw.PlaneMonitor(f"plane{index}", axis="z", position=z, fields=("Ex",)))


def _scene(*, h=H, load=None, load_port=None, probes=True, monitors=False):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-HALF, HALF), (-HALF, HALF), (0.0, LENGTH))),
        grid=mw.GridSpec.uniform(h),
        boundary=mw.BoundarySpec.pec(),
        device="cuda",
    )
    circle = [(B * math.cos(2 * math.pi * k / 96), B * math.sin(2 * math.pi * k / 96)) for k in range(96)]
    square = [(-HALF - h, -HALF - h), (HALF + h, -HALF - h), (HALF + h, HALF + h), (-HALF - h, HALF + h)]
    scene.add_structure(mw.Structure(mw.ComplexPolySlab([square, circle], bounds=(0.0, LENGTH), axis="z"),
                                     mw.Material.pec(), name="shield"))
    scene.add_structure(mw.Structure(mw.Cylinder(position=(0, 0, 0.5 * (Z0 + Z1)), radius=A, height=Z1 - Z0, axis="z"),
                                     mw.Material.pec(), name="inner"))
    scene.add_port(_coax("feed", Z0, reference_impedance=50.0))
    if probes:
        for name, z in PROBES:
            scene.add_port(_coax(name, z))
    if load is not None:
        scene.add_lumped_element(mw.CoaxialLumpedResistor("load", center=(0, 0, Z1), inner_diameter=2 * A,
                                                          outer_diameter=2 * B, normal_axis="z", resistance=load))
    if load_port is not None:
        scene.add_port(load_port)
    if monitors:
        _add_plane_monitors(scene, ROUND_PLANES)
    return scene


def _run(scene, *, steps=6000, source_time=PULSE, frequencies=FREQUENCIES):
    return mw.Simulation.fdtd(
        scene,
        frequencies=frequencies,
        excitations=mw.PortExcitation("feed", source_time=source_time),
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()


def _lattice_impedance(solver, z):
    """TEM impedance ``eta0 / C~`` of the compiled conductor-node sets in the node plane nearest ``z``.

    Conductor nodes are the nodes whose longitudinal Ez edge is shorted (in a uniform
    line both of its endpoints are the same transverse node); the Laplacian runs over
    the free transverse Ex/Ey edges the solver actually updates. Uniform grid: every
    edge has unit conductance (dual width / primal length).
    """
    scene = solver.scene
    x, y, nodes_z = (np.asarray(getattr(scene, f"{axis}_nodes64")) for axis in "xyz")
    k = int(np.argmin(np.abs(nodes_z - z)))
    free_x = solver.cex_curl[:, :, k].cpu().numpy() > 0.0
    free_y = solver.cey_curl[:, :, k].cpu().numpy() > 0.0
    conductor = (solver.cez_curl[:, :, k].cpu().numpy() <= 0.0).ravel()
    xx, yy = np.meshgrid(x, y, indexing="ij")
    inner = conductor & (np.hypot(xx, yy).ravel() < 0.5 * (A + B))
    ids = np.arange(x.size * y.size).reshape(x.size, y.size)
    tails = np.concatenate([ids[:-1][free_x], ids[:, :-1][free_y]])
    heads = np.concatenate([ids[1:][free_x], ids[:, 1:][free_y]])
    edges = np.arange(tails.size)
    incidence = sparse.csr_matrix(
        (np.r_[np.ones(tails.size), -np.ones(tails.size)], (np.r_[edges, edges], np.r_[tails, heads])),
        shape=(tails.size, ids.size),
    )
    laplacian = (incidence.T @ incidence).tocsr()
    phi = inner.astype(np.float64)
    free = ~conductor
    phi[free] = spsolve(laplacian[free][:, free].tocsc(), -(laplacian[free][:, conductor] @ phi[conductor]))
    return ETA0 / float(np.sum((incidence @ phi) ** 2))


@lru_cache(maxsize=None)
def _staircase_impedance(h):
    solver = mw.Simulation.fdtd(_scene(h=h, probes=False).clone(ports=()), frequencies=(FREQUENCIES[0],),
                                run_time=mw.TimeConfig(time_steps=1)).prepare().solver
    return _lattice_impedance(solver, 0.5 * (Z0 + Z1))


def _probe_waves(result, name, z, impedance, h=H):
    """Forward/backward voltage amplitudes from a passive probe.

    The probe voltage sits on its node plane ``z``; its loop current is the H
    circulation half a cell downstream (``direction="+"``), at ``z + h / 2``.
    """
    probe = result.port(name)
    voltage, current = probe.voltage.cpu().numpy(), probe.current.cpu().numpy()
    shifted = z + 0.5 * h
    system = np.array([[np.exp(1j * K0 * z), np.exp(-1j * K0 * z)],
                       [np.exp(1j * K0 * shifted), -np.exp(-1j * K0 * shifted)]]).transpose(2, 0, 1)
    return np.linalg.solve(system, np.stack([voltage, impedance * current], 1)[..., None])[..., 0]


def _line_admittance(waves, z, impedance):
    forward, backward = waves[:, 0] * np.exp(1j * K0 * z), waves[:, 1] * np.exp(-1j * K0 * z)
    return (forward - backward) / (impedance * (forward + backward))


def _line_abcd(impedance):
    c, s = np.cos(K0 * (Z1 - Z0)), np.sin(K0 * (Z1 - Z0))
    return np.array([[c, -1j * impedance * s], [-1j * s / impedance, c]]).transpose(2, 0, 1)


def _shunt(admittance):
    one, zero = np.ones_like(admittance), np.zeros_like(admittance)
    return np.array([[one, zero], [admittance, one]]).transpose(2, 0, 1)


def _two_state_line(states):
    """Transfer matrix between the probes from two independent field states: (beta, Zc, det)."""
    (va1, ia1, vb1, ib1), (va2, ia2, vb2, ib2) = states
    distance = PROBES[1][1] - PROBES[0][1]
    at_a = np.array([[va1, va2], [ia1, ia2]]).transpose(2, 0, 1)
    at_b = np.array([[vb1, vb2], [ib1, ib2]]).transpose(2, 0, 1)
    transfer = at_b @ np.linalg.inv(at_a)
    beta = np.arccos(0.5 * (transfer[:, 0, 0] + transfer[:, 1, 1]).astype(complex)) / distance
    return beta, np.sqrt(transfer[:, 0, 1] / transfer[:, 1, 0]), np.linalg.det(transfer)


def _probe_states(result):
    return tuple(t.cpu().numpy() for name, _z in PROBES for t in (result.port(name).voltage, result.port(name).current))


def _plane_phase_velocity(result, weight, planes):
    """k0 / beta per frequency from a forward + backward fit of a transverse functional over ``planes``."""
    samples = []
    for index in range(len(planes)):
        monitor = result.monitor(f"plane{index}")
        ex = torch.as_tensor(monitor["Ex"]).cpu().numpy()
        x, y = (np.asarray(c) for c in monitor["coords"])
        xx, yy = np.meshgrid(x, y, indexing="ij")
        samples.append((ex * weight(xx, yy)[None]).sum(axis=(1, 2)))
    samples = np.asarray(samples)
    z = np.asarray(planes)
    ratios = []
    for column, k0 in enumerate(K0):
        # Plane fields are independent of the port rows.
        betas = np.linspace(0.8 * k0, 1.2 * k0, 4001)
        residual = []
        for beta in betas:
            basis = np.stack([np.exp(1j * beta * z), np.exp(-1j * beta * z)], 1)
            coefficients = np.linalg.lstsq(basis, samples[:, column], rcond=None)[0]
            residual.append(np.linalg.norm(basis @ coefficients - samples[:, column]))
        ratios.append(k0 / betas[int(np.argmin(residual))])
    return np.asarray(ratios)


@pytest.fixture(scope="module")
def runs():
    """Round coax with an open end (no resistor, the end-capacitance standard) and three loads."""
    return {load: _run(_scene(load=load, monitors=load == 50.0)) for load in (None, 50.0, 100.0, 25.0)}


def _end_admittances(open_line, impedance):
    """Shunt admittances of the feed and load conductor ends, from an open standard and probe_a."""
    waves = _probe_waves(open_line, *PROBES[0], impedance)
    feed = open_line.port("feed")
    feed_admittance = (feed.current / feed.voltage).cpu().numpy() - _line_admittance(waves, Z0, impedance)
    return feed_admittance, _line_admittance(waves, Z1, impedance)


def _fed_reflection(ends, impedance, load):
    """Feed S11 of the ideal line (``impedance``, k0) between the two end shunts, the load in parallel."""
    abcd = _shunt(ends[0]) @ _line_abcd(impedance) @ _shunt(ends[1] + 1.0 / load)
    # The load sits inside the cascade as a shunt, so the feed sees Z_in = A / C.
    fed = abcd[:, 0, 0] / abcd[:, 1, 0]
    return (fed - 50.0) / (fed + 50.0)


@pytest.fixture(scope="module")
def ends(runs):
    return _end_admittances(runs[None], _staircase_impedance(H))


def test_staircase_reference_impedance_converges_to_the_nominal_coax():
    impedances = np.asarray([_staircase_impedance(h) for h in (H, H / 2, H / 4)])
    # Measured 56.910 / 54.236 / 52.068 ohm against the nominal 49.940 ohm.
    deficits = impedances - NOMINAL_IMPEDANCE
    # Conductor nodes lie inside or on the bodies: the staircased rod is inscribed in the
    # nominal one and the shield node set lies outside the nominal shield, so the gap is
    # wider and Z_ref exceeds the nominal value at every level.
    assert np.all(deficits > 0.0), impedances
    # A staircased curved surface converges at first order (deficit ratios 0.62 / 0.50).
    assert np.all(deficits[1:] / deficits[:-1] <= 0.7), deficits
    # First-order extrapolation from the two finest levels: 49.90 ohm, within 1 % of nominal.
    assert abs(2.0 * impedances[2] - impedances[1] - NOMINAL_IMPEDANCE) <= 0.01 * NOMINAL_IMPEDANCE


def test_round_coax_port_launches_a_tem_wave_at_the_speed_of_light(runs):
    def radial(xx, yy):
        radius2 = xx * xx + yy * yy
        return np.where((radius2 > A * A) & (radius2 < B * B), xx / np.maximum(radius2, 1e-30), 0.0)

    # Phase velocity equals c within 0.1 % (measured 0.99989 / 0.99996 / 1.00000; the former
    # two-node-average staircase rule gave 0.877 / 0.872 / 0.868 on this rig).
    assert np.all(np.abs(_plane_phase_velocity(runs[50.0], radial, ROUND_PLANES) - 1.0) <= 1.0e-3)


def test_passive_probes_measure_the_staircase_line(runs):
    beta, impedance, det = _two_state_line((_probe_states(runs[50.0]), _probe_states(runs[100.0])))
    reference = _staircase_impedance(H)
    # Zc = Z_ref within 0.1 % (measured 56.9106 vs 56.9102 ohm, 1e-5).
    assert np.all(np.abs(impedance / reference - 1.0) <= 1.0e-3), (impedance, reference)
    # beta = k0 within 1e-4 (measured 5e-6), lossless reciprocal section (det T = 1).
    assert np.all(np.abs(beta / K0 - 1.0) <= 1.0e-4), beta / K0
    assert np.all(np.abs(det - 1.0) <= 1.0e-4), det


def test_probe_line_impedance_tracks_the_staircase_reference_under_refinement(runs):
    def single_state(result):
        # One state, beta = k0: the two ABCD relations of the probe section give Zc twice.
        va, ia, vb, ib = _probe_states(result)
        distance = PROBES[1][1] - PROBES[0][1]
        c, s = np.cos(K0 * distance), np.sin(K0 * distance)
        return np.sqrt((vb - va * c) / (1j * ia * s) * (1j * va * s) / (ib - ia * c)).real

    coarse = single_state(runs[50.0])
    fine = single_state(_run(_scene(h=H / 2, load=50.0), steps=12000))
    # Measured 56.911 ohm at 0.25 mm and 54.237 ohm at 0.125 mm against Z_ref 56.910 / 54.236.
    for measured, h in ((coarse, H), (fine, H / 2)):
        assert np.all(np.abs(measured / _staircase_impedance(h) - 1.0) <= 1.0e-3), (measured, h)
    assert np.all(np.abs(fine - NOMINAL_IMPEDANCE) < np.abs(coarse - NOMINAL_IMPEDANCE))


def test_open_coaxial_line_ends_are_lossless_lumped_capacitances(ends):
    omega = 2 * math.pi * np.asarray(FREQUENCIES)
    for admittance in ends:
        capacitance = -admittance.imag / omega
        # A pure capacitor: frequency-independent within 1 % (measured 0.2 %), no conductance.
        assert np.all(capacitance > 0.0)
        assert np.ptp(capacitance) <= 0.01 * capacitance.mean(), capacitance
        assert np.all(np.abs(admittance.real) <= 1.0e-3 * np.abs(admittance.imag)), admittance
    feed, load = (-admittance.imag / omega for admittance in ends)
    # The feed face sits 1 mm from its PEC wall, the load face 2 mm: measured 61.5 vs 50.5 fF.
    assert np.all(feed > load)


@pytest.mark.parametrize("load", [50.0, 100.0, 25.0])
def test_coaxial_resistor_adds_exactly_its_conductance_to_the_line_end(runs, ends, load):
    waves = _probe_waves(runs[load], *PROBES[0], _staircase_impedance(H))
    added = _line_admittance(waves, Z1, _staircase_impedance(H)) - ends[1]
    # (Y_L - Y_open) R = 1 within 1e-3 (measured 1 + 3e-5).
    assert np.all(np.abs(added * load - 1.0) <= 1.0e-3), added * load


@pytest.mark.parametrize("load", [50.0, 100.0, 25.0])
def test_coax_fed_loads_match_the_staircase_line_with_its_end_capacitances(runs, ends, load):
    """S11 against ideal line theory with Z_ref, k0 and the two end capacitances.

    The 50 ohm termination is not matched to this line: its 56.910 ohm characteristic
    impedance reflects |S11| 0.023 / 0.048 / 0.075 (bare line theory without the end
    capacitances would give 0.040 / 0.076 / 0.104).
    """
    predicted = _fed_reflection(ends, _staircase_impedance(H), load)
    s11 = runs[load].port("feed").reflection_coefficient.cpu().numpy()
    # |dS11| <= 1e-3 (measured <= 4e-5) with no fitted parameter.
    assert np.all(np.abs(s11 - predicted) <= 1.0e-3), np.abs(s11 - predicted)


def test_coaxial_resistor_equals_a_terminated_coaxial_port_bitwise():
    resistor = _run(_scene(load=80.0, probes=False), steps=1500)
    terminated = _run(_scene(load_port=_coax("end", Z1, reference_impedance=80.0, termination=mw.SeriesRLC(r=80.0)),
                             probes=False), steps=1500)
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(resistor.solver, component), getattr(terminated.solver, component))
    assert torch.equal(resistor.port("feed").voltage, terminated.port("feed").voltage)
    assert torch.equal(resistor.port("feed").current, terminated.port("feed").current)
    end = terminated.port("end")
    # The terminated port observes its own 80 ohm branch: V / I = -80 (current enters the field).
    assert torch.allclose(end.voltage / end.current, torch.full_like(end.voltage, -80.0), rtol=1e-3)


def test_coaxial_port_sweep_matches_the_staircase_line_two_port(ends):
    scene = _scene(load_port=_coax("end", Z1, direction="-", reference_impedance=50.0), probes=False)
    result = mw.Simulation.fdtd(
        scene,
        frequencies=FREQUENCIES,
        excitations=mw.PortSweep(source_time=PULSE),
        run_time=mw.TimeConfig(time_steps=6000),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    s = result.network.s.cpu().numpy()
    assert result.network.port_names == ("feed", "end")
    # The end port shares the resistor's annulus and conductor end (bitwise test above).
    abcd = _shunt(ends[0]) @ _line_abcd(_staircase_impedance(H)) @ _shunt(ends[1])
    a, b, c, d = abcd[:, 0, 0], abcd[:, 0, 1] / 50.0, abcd[:, 1, 0] * 50.0, abcd[:, 1, 1]
    total = a + b + c + d
    predicted = np.stack([np.stack([(a + b - c - d) / total, 2.0 * (a * d - b * c) / total], -1),
                          np.stack([2.0 / total, (-a + b - c + d) / total], -1)], -2)
    # Every entry within 1e-3 of the cascade, reciprocal within 1e-3.
    assert np.all(np.abs(s - predicted) <= 1.0e-3), np.abs(s - predicted)
    assert np.all(np.abs(s[:, 1, 0] - s[:, 0, 1]) <= 1.0e-3)


def _square_scene(load=None, *, probes=True, monitors=False):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-HALF, HALF), (-HALF, HALF), (0.0, LENGTH))),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.pec(),
        device="cuda",
    )
    square = [(-HALF - H, -HALF - H), (HALF + H, -HALF - H), (HALF + H, HALF + H), (-HALF - H, HALF + H)]
    hole = [(-1.75e-3, -1.75e-3), (1.75e-3, -1.75e-3), (1.75e-3, 1.75e-3), (-1.75e-3, 1.75e-3)]
    scene.add_structure(mw.Structure(mw.ComplexPolySlab([square, hole], bounds=(0.0, LENGTH), axis="z"),
                                     mw.Material.pec(), name="shield"))
    scene.add_structure(mw.Structure(mw.Box(position=(0, 0, 0.5 * (Z0 + Z1)), size=(2e-3, 2e-3, Z1 - Z0)),
                                     mw.Material.pec(), name="inner"))
    # The annulus spans both discrete square conductors; the harmonic row adapts to them.
    scene.add_port(_coax("feed", Z0, reference_impedance=50.0))
    if probes:
        for name, z in PROBES:
            scene.add_port(_coax(name, z))
    if load is not None:
        scene.add_lumped_element(mw.CoaxialLumpedResistor("load", center=(0, 0, Z1), inner_diameter=2 * A,
                                                          outer_diameter=2 * B, normal_axis="z", resistance=load))
    if monitors:
        _add_plane_monitors(scene, SQUARE_PLANES)
    return scene


@lru_cache(maxsize=None)
def _square_impedance():
    solver = mw.Simulation.fdtd(_square_scene(probes=False).clone(ports=()), frequencies=(FREQUENCIES[0],),
                                run_time=mw.TimeConfig(time_steps=1)).prepare().solver
    return _lattice_impedance(solver, 0.5 * (Z0 + Z1))


@pytest.fixture(scope="module")
def square_runs():
    return {load: _run(_square_scene(load, monitors=load == 50.0)) for load in (None, 50.0, 25.0, 100.0)}


def test_square_coax_port_launches_a_tem_wave_at_the_speed_of_light(square_runs):
    def sign_mask(xx, yy):
        mask = (np.abs(xx) > 1.1e-3) & (np.abs(xx) < 1.65e-3) & (np.abs(yy) < 0.6e-3)
        return np.where(mask, np.sign(xx), 0.0)

    # Phase velocity equals c within 0.2 % (measured 1.0000).
    assert np.all(np.abs(_plane_phase_velocity(square_runs[50.0], sign_mask, SQUARE_PLANES) - 1.0) <= 2.0e-3)


def test_square_coax_line_and_loads_match_the_lattice_reference(square_runs):
    reference = _square_impedance()
    beta, impedance, _det = _two_state_line((_probe_states(square_runs[50.0]), _probe_states(square_runs[100.0])))
    # Lattice reference 28.448 ohm; probes measure it within 1e-7 and beta = k0 within 1e-5.
    assert np.all(np.abs(impedance / reference - 1.0) <= 1.0e-3), (impedance, reference)
    assert np.all(np.abs(beta / K0 - 1.0) <= 1.0e-4), beta / K0
    ends = _end_admittances(square_runs[None], reference)
    for load in (50.0, 25.0, 100.0):
        s11 = square_runs[load].port("feed").reflection_coefficient.cpu().numpy()
        # |dS11| <= 1e-3 (measured <= 2e-5) with no fitted parameter. A single fitted line
        # impedance without the end capacitances settled at 26.9 ohm, 5 % low.
        assert np.all(np.abs(s11 - _fed_reflection(ends, reference, load)) <= 1.0e-3), load
