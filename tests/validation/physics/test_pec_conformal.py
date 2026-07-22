"""Field-level contract for conformal PEC (``SubpixelSpec(pec="conformal")``).

The conformal open fraction multiplies the electric update every step, so a fill
``f`` on an edge is an effective conductivity ``eps*f/dt`` there. Two consequences
define the contract these tests lock in:

1. A conductor face parallel to the grid cuts no tangential edge, so conformal must
   reproduce the staircase result exactly. Anything else means fractional fill has
   leaked onto edges the surface never reaches, which paints a lossy shell around
   the metal -- the defect that made a grid-aligned PEC slab an order of magnitude
   worse than staircase against an external reference.
2. On a genuinely cut edge the soft short *is* lossy. A PEC scatterer is lossless,
   so that spurious absorption is a bounded, measured approximation error rather
   than a physical effect; the cavity gate below pins how much of it is tolerated
   and would catch a regression back to the much lossier smoothed-occupancy path.

Sub-cell placement of a *flat, grid-parallel* wall is deliberately not claimed: it
requires the area-scaled (Dey-Mittra) magnetic update, not an electric-edge factor.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw


C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
MU0 = 4.0e-7 * np.pi

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Conformal PEC field validation needs CUDA"
)


def _peak_frequency(signal, dt):
    signal = signal - signal.mean()
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, dt)
    band = (freqs > 5e7) & (freqs < 1.2e9)
    idx = np.where(band)[0][np.argmax(spectrum[band])]
    a, b, c = spectrum[idx - 1], spectrum[idx], spectrum[idx + 1]
    curvature = a - 2 * b + c
    delta = 0.5 * (a - c) / curvature if curvature != 0 else 0.0
    return freqs[idx] + delta * (freqs[1] - freqs[0])


def _cavity_resonance(wall_x, pec_mode, *, grid=0.01, length=0.6, steps=12000):
    domain = mw.Domain(bounds=((0.0, length), (0.0, 0.05), (0.0, 0.05)))
    scene = mw.Scene(
        domain=domain,
        grid=mw.GridSpec.uniform(grid),
        boundary=mw.BoundarySpec.periodic().with_faces(x_low="pec", x_high="pec"),
        device="cuda",
        subpixel_samples=mw.SubpixelSpec(pec=pec_mode),
    )
    scene.add_structure(
        mw.Structure(
            name="wall",
            geometry=mw.Box(
                position=((wall_x + length) / 2.0, 0.025, 0.025),
                size=(length - wall_x, 0.2, 0.2),
            ),
            material=mw.Material.pec(),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(wall_x * 0.5, 0.025, 0.025),
            width=grid * 2,
            polarization=(0, 0, 1),
            source_time=mw.RickerWavelet(frequency=5e8, amplitude=1.0),
            name="src",
        )
    )
    scene.add_monitor(
        mw.FieldTimeMonitor(position=(wall_x * 0.27, 0.025, 0.025), name="probe", components=("Ez",))
    )
    result = mw.Simulation.fdtd(
        scene, frequencies=(5e8,), run_time=mw.TimeConfig(time_steps=steps)
    ).run()
    signal = result.monitor("probe")["data"].detach().cpu().numpy().reshape(-1)
    return _peak_frequency(signal, result.solver.dt)


def test_conformal_pec_reproduces_staircase_for_a_grid_parallel_wall():
    """A wall normal to x cuts no tangential edge anywhere inside the cell."""
    # 0.30 lands exactly on a node plane; the others sit at 20%, 50% and 80% of the
    # way across one 10 mm cell, so any fill leaking off the cut edges would show.
    for wall in (0.30, 0.302, 0.305, 0.308):
        staircase = _cavity_resonance(wall, "staircase")
        conformal = _cavity_resonance(wall, "conformal")
        relative = abs(conformal - staircase) / staircase
        # Measured: 1.8e-14 (aligned) to 1.8e-10 (mid-cell).
        assert relative < 1e-8, (wall, staircase, conformal, relative)


def _slab_probe(pec_mode, *, dx=0.02, steps=1500):
    """Ex(t) just outside a node-aligned PEC slab hit by a plane wave."""
    half = 0.3
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=8).with_faces(
            x_low="periodic", x_high="periodic", y_low="periodic", y_high="periodic"
        ),
        subpixel_samples=mw.SubpixelSpec(pec=pec_mode),
        device="cuda",
    )
    # The slab faces at +-0.06 are exact multiples of the 20 mm spacing, so the
    # staircase representation of this geometry is exact and conformal must match it.
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(4.0, 4.0, 0.12)),
            material=mw.Material.pec(),
            name="slab",
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0, 0, 1),
            polarization="Ex",
            source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=0.7e9),
        )
    )
    scene.add_monitor(
        mw.FieldTimeMonitor("probe", components=("Ex",), position=(0.0, 0.0, -0.16))
    )
    result = mw.Simulation.fdtd(
        scene, frequencies=(2.0e9,), run_time=mw.TimeConfig(time_steps=steps)
    ).run()
    return result.monitor("probe")["data"].detach().cpu().numpy().reshape(-1)


def test_axis_aligned_pec_slab_reflection_is_identical_under_conformal():
    staircase = _slab_probe("staircase")
    conformal = _slab_probe("conformal")
    assert np.max(np.abs(staircase)) > 0.0
    assert np.array_equal(staircase, conformal)


def _cavity_energy(pec_mode, steps, *, dx=0.02, half=0.2):
    """Total EM energy left in a closed PEC cavity holding a PEC sphere."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pec(),
        subpixel_samples=mw.SubpixelSpec(pec=pec_mode),
        device="cuda",
    )
    # Off-centre so the sphere surface cuts edges asymmetrically in all three axes.
    scene.add_structure(
        mw.Structure(
            geometry=mw.Sphere(position=(0.01, 0.0, -0.013), radius=0.083),
            material=mw.Material.pec(),
            name="sphere",
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.12, 0.09, 0.11),
            polarization=(0, 0, 1),
            source_time=mw.RickerWavelet(frequency=4.0e9, amplitude=1.0),
            name="src",
        )
    )
    scene.add_monitor(
        mw.FieldTimeMonitor("probe", components=("Ez",), position=(-0.1, -0.08, 0.06))
    )
    prepared = mw.Simulation.fdtd(
        scene, frequencies=(4.0e9,), run_time=mw.TimeConfig(time_steps=steps)
    ).prepare()
    prepared.run()
    solver = prepared.solver
    electric = sum(
        float((getattr(solver, component).float() ** 2).sum()) for component in ("Ex", "Ey", "Ez")
    )
    magnetic = sum(
        float((getattr(solver, component).float() ** 2).sum()) for component in ("Hx", "Hy", "Hz")
    )
    return 0.5 * EPS0 * electric + 0.5 * MU0 * magnetic


def test_closed_pec_cavity_with_a_curved_conductor_is_lossless_under_staircase():
    early = _cavity_energy("staircase", 800)
    late = _cavity_energy("staircase", 6000)
    assert early > 0.0
    # Vacuum plus PEC is lossless; measured retention 0.999989.
    assert abs(late / early - 1.0) < 1e-3


def test_conformal_pec_spurious_absorption_on_a_curved_conductor_stays_bounded():
    """Conformal's soft short is lossy on cut edges; bound how lossy.

    Measured energy retained after 6000 steps (5200 of them source-free):
    staircase 0.999989, conformal 0.450052, and the pre-fix smoothed-occupancy
    conformal 0.124665. The bound sits between the last two, so a regression back
    to the node-averaged fill (which also leaks fill onto vacuum edges) fails here.
    """
    early = _cavity_energy("conformal", 800)
    late = _cavity_energy("conformal", 6000)
    assert early > 0.0
    retained = late / early
    assert retained > 0.30
    assert retained < 1.0 + 1e-3
