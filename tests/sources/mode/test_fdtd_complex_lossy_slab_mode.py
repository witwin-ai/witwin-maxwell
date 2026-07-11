"""Complex (lossy) waveguide mode acceptance test.

The FDTD scalar mode solver must return the complex effective index of a lossy
symmetric slab waveguide to within 1e-3 of an independent reference. The
reference is the analytic complex ``n_eff`` of the fundamental even TE mode,
obtained by solving the symmetric-slab transcendental dispersion equation with
SciPy on the *same* complex core permittivity the solver sees.

Geometry. The slab confines along ``y`` (finely resolved) and is uniform along
``x``. For the ``Ex`` (TE) polarization the transverse mode operator reduces
exactly to the 1-D slab Helmholtz equation ``d2E/dy2 + (k0**2 eps(y) - beta**2) E
= 0``, so the finite-difference eigenvalue ``beta**2`` matches the transcendental
slab mode. ``x`` is discretized with four wide cells so the single interior
``x_half`` node carries negligible spurious transverse momentum, and the core
interface is aligned to a ``y``-node midpoint so the sampled step slab matches the
analytic one to second order.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import brentq, newton

import witwin.maxwell as mw
from witwin.core.material import VACUUM_PERMITTIVITY


def _analytic_even_te_neff(eps_core: complex, eps_clad: float, k0: float, a: float) -> complex:
    """Fundamental even TE complex ``n_eff`` of a symmetric slab (half-width ``a``).

    Even TE dispersion: ``gamma = kappa * tan(kappa * a)`` with the transverse
    core wavenumber ``kappa = k0 sqrt(eps_core - n**2)`` and cladding decay
    ``gamma = k0 sqrt(n**2 - eps_clad)``. The pole-free residual
    ``kappa_amp sin(U) - gamma_amp cos(U)`` (``U = k0 a kappa_amp``) is bracketed
    on the real (lossless) problem, then continued to the complex root with a
    secant iteration.
    """

    def residual(n):
        kappa = np.sqrt(eps_core - n * n)
        gamma = np.sqrt(n * n - eps_clad)
        u = k0 * a * kappa
        return kappa * np.sin(u) - gamma * np.cos(u)

    n_core = float(np.sqrt(np.real(eps_core)))
    n_clad = float(np.sqrt(eps_clad))
    scan = np.linspace(n_core - 1e-6, n_clad + 1e-6, 4000)
    values = [np.real(residual(complex(n))) for n in scan]
    real_root = None
    for i in range(len(scan) - 1):
        if values[i] == 0.0:
            real_root = float(scan[i])
            break
        if values[i] * values[i + 1] < 0.0:
            real_root = brentq(lambda n: float(np.real(residual(complex(n)))), scan[i], scan[i + 1])
            break
    if real_root is None:
        raise RuntimeError("no fundamental even TE root found for the lossless slab")
    return complex(newton(residual, complex(real_root, 0.0), tol=1e-13, maxiter=200))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_lossy_slab_complex_effective_index_matches_transcendental_solver():
    f = 1.0e9
    c = 299792458.0
    lam = c / f

    n_core = 1.5
    eps_clad = 1.0
    # Loss: sigma_e chosen so Im(eps_core) = -0.1 at f, in the same eps = eps' -
    # i sigma/(omega eps0) convention the compiled material model uses.
    im_eps = 0.1
    sigma_e = im_eps * (2.0 * np.pi * f) * VACUUM_PERMITTIVITY
    core = mw.Material(eps_r=n_core * n_core, sigma_e=sigma_e, name="core")
    eps_core = core.relative_permittivity(f)

    # Confinement grid (y): fine, symmetric about the y = 0 node.
    dy = lam / 80.0
    # Slab half-width snapped to a node midpoint a = (K + 0.5) dy so the sampled
    # step interface coincides with the analytic one.
    k_half = int(round(0.25 * lam / dy - 0.5))
    a = (k_half + 0.5) * dy
    d = 2.0 * a
    clad = 3.0 * lam
    ly = d + 2.0 * clad
    ny_cells = int(round(ly / dy))
    if ny_cells % 2 == 1:
        ny_cells += 1
    ly = ny_cells * dy
    y0 = -0.5 * ly

    # Uniform axis (x): four wide cells. Ex lives on x_half (3 points) -> a single
    # interior transverse node whose Dirichlet momentum ~ 2/dx**2 is negligible.
    dx = 20.0 * lam
    ap_x = 2.8 * dx  # spans all four x-nodes and all three x_half points

    dz = lam / 20.0
    nz_cells = 40
    lz = nz_cells * dz

    x_coords = np.array([-1.5 * dx, -0.5 * dx, 0.5 * dx, 1.5 * dx], dtype=np.float64)
    y_coords = np.array([y0 + i * dy for i in range(ny_cells + 1)], dtype=np.float64)
    z_coords = np.array([-0.5 * lz + i * dz for i in range(nz_cells + 1)], dtype=np.float64)

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.5 * dx, 1.5 * dx), (y0, y0 + ly), (-0.5 * lz, 0.5 * lz))),
        grid=mw.GridSpec.custom(x_coords, y_coords, z_coords),
        boundary=mw.BoundarySpec(kind="none", num_layers=8, strength=1e6, z="pml"),
        device="cuda",
    )
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, 0.0), size=(6.0 * dx, d, 2.0 * lz)).with_material(core, name="slab")
    )
    scene.add_source(
        mw.ModeSource(
            position=(0.0, 0.0, 0.0),
            size=(ap_x, ly - 4.0 * dy, 0.0),
            polarization="Ex",
            source_time=mw.CW(frequency=f, amplitude=1.0),
            name="mode0",
        )
    )

    solver = mw.Simulation.fdtd(scene, frequencies=[f]).prepare().solver
    compiled = solver._compiled_sources[0]

    # The lossy plane must be routed to the complex scalar eigen-solve.
    assert compiled["mode_solver_kind"] == "scalar_complex_dense"
    n_sim = compiled["effective_index_complex"]
    assert n_sim is not None

    k0 = 2.0 * np.pi * f / float(solver.c)
    n_ana = _analytic_even_te_neff(eps_core, eps_clad, k0, a)

    # A genuinely complex, guided mode: the loss must show up as a clear negative
    # imaginary part, not numerical noise.
    assert n_ana.imag < -1.0e-2
    assert n_sim.imag < -1.0e-2
    assert n_core > n_sim.real > np.sqrt(eps_clad)

    # Acceptance criterion for the lossy-mode item.
    assert abs(n_sim - n_ana) < 1.0e-3
