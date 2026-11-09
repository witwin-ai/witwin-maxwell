"""TEM phase velocity of vacuum coax lines with curved or multi-box PEC conductors.

A Yee line whose shorted PEC edges are the edge set of a node set has an exact
discrete TEM null space: its TEM wave sees only the 1D Yee dispersion along the
line, the same for every conductor cross section. A grid-aligned square coax and
a round (cylinder / polygon) coax must therefore propagate at the same speed.
Before the node-consistent staircase rule the round coax ran at 0.928c and a
union of node-centred boxes at 0.833c (0.25 mm cells, 5-11 GHz); see
docs/memory/rf-coax-curved-pec-slow-wave.md.

The wave is launched by four radial dipoles (they excite only the TEM and
azimuthal-order-4 families; the latter is cut off far above 11 GHz) and measured
on 13 Ex/Ey planes. k is fitted with a forward+backward two-wave model, so a
small PML reflection does not bias it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="coax TEM velocity runs need CUDA")

C0 = 299_792_458.0
A = 1.0e-3
B = 2.3e-3
SQUARE_A, SQUARE_B = 1.0e-3, 1.75e-3
HALF = 3.0e-3
LENGTH = 30.0e-3
LAYERS = 12
FREQUENCIES = (5.0e9, 8.0e9, 11.0e9)
PLANES = tuple(12.0e-3 + 1.0e-3 * i for i in range(13))


def _frame(h):
    return [(-HALF - h, -HALF - h), (HALF + h, -HALF - h), (HALF + h, HALF + h), (-HALF - h, HALF + h)]


def _node_centred_boxes(h, z0, z1):
    n = int(math.ceil(A / h))
    return [mw.Box(position=(i * h, j * h, 0.5 * (z0 + z1)), size=(h, h, z1 - z0))
            for i in range(-n, n + 1) for j in range(-n, n + 1) if (i * h) ** 2 + (j * h) ** 2 <= A * A]


def _abutting_slabs(h, z0, z1):
    boxes = []
    for j in range(-int(math.ceil(A / h)), int(math.ceil(A / h))):
        yc = (j + 0.5) * h
        half = math.floor(math.sqrt(max(A * A - yc * yc, 0.0)) / h) * h
        if half > 0:
            boxes.append(mw.Box(position=(0.0, yc, 0.5 * (z0 + z1)), size=(2 * half, h, z1 - z0)))
    return boxes


def _coax(case, h, pec_mode="staircase"):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-HALF, HALF), (-HALF, HALF), (0.0, LENGTH))),
        grid=mw.GridSpec.uniform(h),
        # Conductors run through the z PML so the line is matched at both ends.
        boundary=mw.BoundarySpec.faces(default="pec", num_layers=LAYERS, strength=1.0, z="pml"),
        subpixel_samples=mw.SubpixelSpec(pec=pec_mode, averaging="arithmetic", samples=1),
        device="cuda",
    )
    z0, z1 = -(LAYERS + 2) * h, LENGTH + (LAYERS + 2) * h
    if case == "square":
        hole = [(-SQUARE_B, -SQUARE_B), (SQUARE_B, -SQUARE_B), (SQUARE_B, SQUARE_B), (-SQUARE_B, SQUARE_B)]
        inner = [mw.Box(position=(0, 0, 0.5 * (z0 + z1)), size=(2 * SQUARE_A, 2 * SQUARE_A, z1 - z0))]
        outer_radius = SQUARE_B
    else:
        vertices = 96 if h >= 0.25e-3 else 256
        hole = [(B * math.cos(2 * math.pi * k / vertices), B * math.sin(2 * math.pi * k / vertices))
                for k in range(vertices)]
        outer_radius = B
        inner = {
            "cylinder": lambda: [mw.Cylinder(position=(0, 0, 0.5 * (z0 + z1)), radius=A, height=z1 - z0, axis="z")],
            "node_centred_boxes": lambda: _node_centred_boxes(h, z0, z1),
            "abutting_slabs": lambda: _abutting_slabs(h, z0, z1),
        }[case]()
    scene.add_structure(mw.Structure(mw.ComplexPolySlab([_frame(h), hole], bounds=(z0, z1), axis="z"),
                                     mw.Material.pec(), name="shield"))
    for index, geometry in enumerate(inner):
        scene.add_structure(mw.Structure(geometry, mw.Material.pec(), name=f"inner{index}"))
    radius = 0.5 * (A + outer_radius)
    pulse = mw.GaussianPulse(frequency=9.0e9, fwidth=6.0e9)
    for index, (cx, cy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        scene.add_source(mw.PointDipole(position=(cx * radius, cy * radius, 4.0e-3), width=2 * h,
                                        polarization=(cx, cy, 0), source_time=pulse, name=f"src{index}"))
    for index, z in enumerate(PLANES):
        for component in ("Ex", "Ey"):
            scene.add_monitor(mw.PlaneMonitor(f"p{index}{component}", axis="z", position=z,
                                              fields=(component,), frequencies=FREQUENCIES))
    return scene, outer_radius


def _two_wave_k(z, values, k0):
    z = np.asarray(z) - np.mean(z)

    def residual(k):
        basis = np.stack([np.exp(1j * k * z), np.exp(-1j * k * z)], axis=1)
        coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
        return np.linalg.norm(basis @ coefficients - values) / np.linalg.norm(values)

    grid = np.linspace(0.7 * k0, 1.3 * k0, 1201)
    best = int(np.argmin([residual(k) for k in grid]))
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]
    for _ in range(80):
        first, second = low + (high - low) / 3, high - (high - low) / 3
        if residual(first) < residual(second):
            high = second
        else:
            low = first
    k = 0.5 * (low + high)
    return k, residual(k)


def _phase_velocity(case, h, pec_mode="staircase"):
    scene, outer_radius = _coax(case, h, pec_mode)
    result = mw.Simulation.fdtd(
        scene, absorber="cpml",
        frequencies=list(FREQUENCIES),
        run_time=mw.TimeConfig(time_steps=int(round(6000 * 0.25e-3 / h))),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()
    functional = []
    for index in range(len(PLANES)):
        total = 0.0
        for component, axis in (("Ex", 0), ("Ey", 1)):
            monitor = result.monitor(f"p{index}{component}")
            grids = np.meshgrid(*monitor["coords"], indexing="ij")
            radius = np.hypot(grids[0], grids[1])
            weight = np.where((radius > A) & (radius < outer_radius), grids[axis] / np.maximum(radius, 1e-12) ** 2, 0.0)
            total = total + np.tensordot(monitor["data"].detach().cpu().numpy(), weight, axes=([-2, -1], [0, 1]))
        functional.append(total)
    functional = np.asarray(functional)
    velocities = []
    for column, frequency in enumerate(FREQUENCIES):
        k0 = 2 * math.pi * frequency / C0
        k, residual = _two_wave_k(PLANES, functional[:, column], k0)
        assert residual < 1.0e-4, (case, frequency, residual)  # a clean single-mode line
        velocities.append(k0 / k)
    return np.asarray(velocities)


@pytest.fixture(scope="module")
def square_reference():
    return _phase_velocity("square", 0.25e-3)


def test_grid_aligned_square_coax_runs_at_c(square_reference):
    # Only the 1D Yee dispersion along z remains: measured 0.99998 / 0.99995 / 0.99990.
    assert np.all(np.abs(square_reference - 1.0) <= 2.0e-4), square_reference


@pytest.mark.parametrize("case", ["cylinder", "node_centred_boxes", "abutting_slabs"])
def test_curved_and_multibox_coax_tem_velocity(case, square_reference):
    velocity = _phase_velocity(case, 0.25e-3)
    # Stated gate: |v/c - 1| <= 0.5 % at 0.25 mm (before the fix: 0.928 cylinder,
    # 0.833 node-centred boxes). Sharper: the consistent node set reproduces the
    # aligned-square line to fit precision (measured |dv| < 1e-6).
    assert np.all(np.abs(velocity - 1.0) <= 5.0e-3), velocity
    assert np.all(np.abs(velocity - square_reference) <= 2.0e-5), (velocity, square_reference)


def test_round_coax_residual_dispersion_converges_at_second_order(square_reference):
    coarse = _phase_velocity("cylinder", 0.25e-3)
    fine = _phase_velocity("cylinder", 0.125e-3)
    deficit_coarse, deficit_fine = 1.0 - coarse, 1.0 - fine
    assert np.all(deficit_coarse > 0.0) and np.all(deficit_fine > 0.0)
    # Yee dispersion (kh)^2: halving h quarters the deficit; the former slot-loaded
    # line roughly halved it (0.0735 -> 0.0457 at 5 GHz). Measured ratios 0.36 / 0.23
    # / 0.25; the 5 GHz deficit (2e-5) is closest to the fit floor, so the tight
    # second-order band is asserted at 11 GHz.
    ratio = deficit_fine / deficit_coarse
    assert np.all(ratio < 0.4), ratio
    assert 0.2 < ratio[-1] < 0.3, ratio
    assert np.all(deficit_coarse < 2.0e-4), deficit_coarse
