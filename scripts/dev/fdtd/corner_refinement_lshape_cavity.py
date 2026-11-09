"""Corner refinement physics check: TM resonance of an L-shaped PEC cavity.

The cavity cross-section is ``[-a, a]^2`` minus the quadrant ``[0, a] x [-a, 0]``
(three squares of side ``a``), filled by a PEC block whose convex corner at the
origin is the cavity's re-entrant corner. The z-invariant TM modes (Ez only)
satisfy the 2D Dirichlet Laplacian, whose first eigenvalue is
``lambda_1 a^2 = 9.6397238440219`` (Fox-Henrici-Moler / Trefethen-Betcke). The
H field is singular like ``r^(-1/3)`` at the re-entrant corner, so uniform
refinement converges slowly and refinement at that one corner pays off.

The Yee time step error is removed exactly: for a lossless cavity the discrete
frequency obeys ``(2 / (c dt) sin(w dt / 2))^2 = lambda_h``, the eigenvalue of
the spatial Yee operator on the grid. ``lambda_h`` is therefore a pure measure
of spatial discretization quality.

Run (GPU 1): ``python scripts/dev/fdtd/corner_refinement_lshape_cavity.py OUT.json``
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch
from scipy.optimize import least_squares

import witwin.maxwell as mw

A = 0.1
C0 = 299_792_458.0
LAMBDA1 = 9.6397238440219
F1 = C0 * math.sqrt(LAMBDA1) / A / (2.0 * math.pi)
HZ = 0.05  # z-invariant modes; a thick single-cell z keeps dt set by the plane


def _scene(grid, z_nodes=None):
    block = mw.Structure(
        mw.Box(position=(0.75 * A, -0.75 * A, 0.5 * HZ), size=(1.5 * A, 1.5 * A, 4 * HZ)),
        mw.Material.pec(),
        name="block",
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-A, A), (-A, A), (0.0, HZ))),
        grid=grid,
        boundary=mw.BoundarySpec.pec(),
        device="cuda",
        structures=[block],
        sources=[
            mw.PointDipole(
                position=(-0.53 * A, 0.37 * A, 0.5 * HZ),
                polarization="Ez",
                source_time=mw.GaussianPulse(frequency=F1, fwidth=0.25 * F1),
                name="src",
            )
        ],
    )
    scene.add_monitor(mw.FieldTimeMonitor("probe", components=("Ez",), position=(-0.31 * A, 0.62 * A, 0.5 * HZ)))
    return scene


def uniform_grid(n):
    nodes = np.linspace(-A, A, 2 * n + 1)
    return mw.GridSpec.custom(nodes, nodes.copy(), np.array([0.0, HZ]))


def corner_grid(n_base, dl, num_cells, max_ratio=1.4):
    """Coarse auto mesh (``n_base`` steps per ``a`` in vacuum) plus corner refinement."""
    return mw.GridSpec.auto(
        wavelength=A,
        min_steps_per_wavelength=n_base,
        max_ratio=max_ratio,
        layer_refinement=mw.LayerRefinementSpec(
            min_cells=None,
            axis="z",
            bounds=(0.0, HZ),
            corner_finder=mw.CornerFinderSpec(medium="metal"),
            corner_refinement=mw.GridRefinement(dl=dl, num_cells=num_cells),
        ),
    )


def _fit_frequencies(t, signal, guesses):
    """Variable-projection least squares for a sum of undamped sinusoids."""
    t = t - t[0]

    def residual(omegas):
        basis = np.concatenate([np.cos(np.outer(t, omegas)), np.sin(np.outer(t, omegas))], axis=1)
        coeffs, *_ = np.linalg.lstsq(basis, signal, rcond=None)
        return basis @ coeffs - signal

    fit = least_squares(residual, np.asarray(guesses), x_scale=np.asarray(guesses) * 1e-4,
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return fit.x


def measure(grid, *, periods=300):
    scene = _scene(grid)
    prepared = mw.Simulation.fdtd(scene, frequencies=[F1], full_field_dft=False,
                                  run_time=mw.TimeConfig(time_steps=10)).prepare()
    dt = float(prepared.solver.dt)
    steps = int(math.ceil(periods / F1 / dt))
    start = time.perf_counter()
    result = mw.Simulation.fdtd(scene, frequencies=[F1], full_field_dft=False,
                                run_time=mw.TimeConfig(time_steps=steps)).run()
    elapsed = time.perf_counter() - start
    payload = result.monitor("probe")
    t = np.asarray(payload["t"].detach().cpu().numpy() if torch.is_tensor(payload["t"]) else payload["t"], dtype=np.float64)
    ez = payload["data"]
    ez = np.asarray(ez.detach().cpu().numpy() if torch.is_tensor(ez) else ez, dtype=np.float64).reshape(t.size, -1)[:, 0]
    keep = t > 12.0 / (0.25 * F1)  # after the source pulse
    t, ez = t[keep], ez[keep]
    window = np.hanning(t.size)
    spectrum = np.abs(np.fft.rfft(ez * window, 8 * t.size))
    freqs = np.fft.rfftfreq(8 * t.size, d=float(t[1] - t[0]))
    band = (freqs > 0.7 * F1) & (freqs < 1.6 * F1)
    peaks = [i for i in np.flatnonzero(band)[1:-1] if spectrum[i] >= spectrum[i - 1] and spectrum[i] >= spectrum[i + 1]]
    peaks = sorted(peaks, key=lambda i: -spectrum[i])[:4]
    omegas = _fit_frequencies(t, ez, [2 * math.pi * freqs[i] for i in peaks])
    omega1 = min(omegas, key=lambda w: abs(w - 2 * math.pi * F1))
    lam_h = (2.0 / (C0 * dt) * math.sin(0.5 * omega1 * dt)) ** 2
    return {
        "lambda_h_a2": lam_h * A * A,
        "relative_error": lam_h * A * A / LAMBDA1 - 1.0,
        "dt": dt,
        "steps": steps,
        "seconds": elapsed,
    }


def grid_stats(grid):
    from witwin.maxwell.fdtd.meshing import resolve_auto_grid

    scene = _scene(grid).clone(device="cpu")
    if grid.is_auto:
        x, y, _ = resolve_auto_grid(scene)
    else:
        x, y = grid.x_coords, grid.y_coords
    out = {"nx": int(x.size - 1), "ny": int(y.size - 1), "cells_2d": int((x.size - 1) * (y.size - 1))}
    for name, nodes in (("x", x), ("y", y)):
        d = np.diff(nodes)
        out[f"{name}_min_dl"] = float(d.min())
        out[f"{name}_max_ratio"] = float(max((d[1:] / d[:-1]).max(), (d[:-1] / d[1:]).max()))
        out[f"{name}_has_corner_node"] = bool(np.any(np.abs(nodes) < 1e-12))
    return out


def main(path):
    cases = {f"uniform_n{n}": uniform_grid(n) for n in (10, 20, 40, 80)}
    for n_base, dl_div, num in ((10, 4, 8), (10, 8, 16), (20, 8, 16)):
        cases[f"corner_n{n_base}_dl{dl_div}x_c{num}"] = corner_grid(n_base, A / n_base / dl_div, num)
    report = {"lambda1_a2_exact": LAMBDA1, "f1_hz": F1, "cases": {}}
    for name, grid in cases.items():
        entry = grid_stats(grid)
        entry.update(measure(grid))
        report["cases"][name] = entry
        print(name, json.dumps(entry), flush=True)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
