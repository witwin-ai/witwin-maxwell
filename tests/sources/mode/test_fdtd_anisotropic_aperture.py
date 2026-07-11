"""Diagonal-anisotropic waveguide aperture acceptance tests.

An anisotropic aperture must be solved with the permittivity component that
matches the injected polarization (``E_p`` sees ``eps_pp``), the same per-axis
component the forward Yee update uses for that field. Averaging the three
diagonal components (the pre-fix behavior) injects a mode computed for an
isotropic medium the forward solve never sees.

Geometry. A symmetric slab confines along ``y`` (finely resolved) and is uniform
along ``x`` (four wide cells), propagating along ``z``. For the ``Ex`` (TE)
polarization the transverse mode operator decouples into the 1-D slab Helmholtz
``d2E_x/dy2 + (k0**2 eps_xx(y) - beta**2) E_x = 0``, so the fundamental effective
index depends on ``eps_xx`` alone -- not on ``eps_yy``, ``eps_zz``, or their
average. ``x`` is discretized with four wide cells so the single interior
transverse node carries negligible spurious momentum, and the core interface is
aligned to a ``y``-node midpoint so the sampled step slab matches the analytic
one to second order.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import brentq, newton

import witwin.maxwell as mw
from witwin.core.material import VACUUM_PERMITTIVITY


def _analytic_even_te_neff(eps_core: complex, eps_clad: float, k0: float, a: float) -> complex:
    """Fundamental even TE complex ``n_eff`` of a symmetric slab (half-width ``a``)."""

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


def _slab_grid(*, dy_div=80.0, clad_lam=3.0):
    """Shared slab grid (confine ``y``, wide ``x``, PML along ``z``).

    ``dy_div`` sets the ``y`` resolution (``lam / dy_div``) and ``clad_lam`` the
    cladding thickness in wavelengths. The defaults give the finely resolved,
    thickly clad slab the scalar-complex analytic match needs; the full-vector
    test passes a coarser, thinner grid so the aperture stays on the dense
    vector eigen-solve (``<= 384`` interior unknowns) while still resolving the
    guided mode.
    """
    f = 1.0e9
    c = 299792458.0
    lam = c / f

    dy = lam / dy_div
    k_half = int(round(0.25 * lam / dy - 0.5))
    a = (k_half + 0.5) * dy
    d = 2.0 * a
    clad = clad_lam * lam
    ly = d + 2.0 * clad
    ny_cells = int(round(ly / dy))
    if ny_cells % 2 == 1:
        ny_cells += 1
    ly = ny_cells * dy
    y0 = -0.5 * ly

    dx = 20.0 * lam
    ap_x = 2.8 * dx
    dz = lam / 20.0
    nz_cells = 40
    lz = nz_cells * dz

    x_coords = np.array([-1.5 * dx, -0.5 * dx, 0.5 * dx, 1.5 * dx], dtype=np.float64)
    y_coords = np.array([y0 + i * dy for i in range(ny_cells + 1)], dtype=np.float64)
    z_coords = np.array([-0.5 * lz + i * dz for i in range(nz_cells + 1)], dtype=np.float64)
    return {
        "f": f,
        "lam": lam,
        "a": a,
        "d": d,
        "dx": dx,
        "dy": dy,
        "ap_x": ap_x,
        "ly": ly,
        "y0": y0,
        "lz": lz,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
    }


def _build_slab_scene(core_material, grid):
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-1.5 * grid["dx"], 1.5 * grid["dx"]),
                (grid["y0"], grid["y0"] + grid["ly"]),
                (-0.5 * grid["lz"], 0.5 * grid["lz"]),
            )
        ),
        grid=mw.GridSpec.custom(grid["x_coords"], grid["y_coords"], grid["z_coords"]),
        boundary=mw.BoundarySpec(kind="none", num_layers=8, strength=1e6, z="pml"),
        device="cuda",
    )
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, 0.0), size=(6.0 * grid["dx"], grid["d"], 2.0 * grid["lz"])).with_material(
            core_material, name="slab"
        )
    )
    scene.add_source(
        mw.ModeSource(
            position=(0.0, 0.0, 0.0),
            size=(grid["ap_x"], grid["ly"] - 4.0 * grid["dy"], 0.0),
            polarization="Ex",
            source_time=mw.CW(frequency=grid["f"], amplitude=1.0),
            name="mode0",
        )
    )
    return scene


def _mode_effective_index(core_material, grid):
    scene = _build_slab_scene(core_material, grid)
    solver = mw.Simulation.fdtd(scene, frequencies=[grid["f"]]).prepare().solver
    return solver._compiled_sources[0]


# Anisotropic core with eps_xx the largest (so the Ex-polarized TE mode is the
# fundamental) and distinct eps_yy, eps_zz. The isotropic average differs
# strongly from eps_xx, which is what a correct anisotropic solve must not use.
_EPS_XX = 6.25
_EPS_YY = 2.25
_EPS_ZZ = 4.0
_EPS_AVG = (_EPS_XX + _EPS_YY + _EPS_ZZ) / 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_vector_anisotropic_aperture_uses_polarization_component():
    """The full-vector solve resolves eps_xx for the Ex mode, not the average."""
    grid = _slab_grid(dy_div=40.0, clad_lam=1.0)

    aniso = _mode_effective_index(
        mw.Material(epsilon_tensor=mw.DiagonalTensor3(_EPS_XX, _EPS_YY, _EPS_ZZ)), grid
    )
    iso_xx = _mode_effective_index(mw.Material(eps_r=_EPS_XX), grid)
    iso_avg = _mode_effective_index(mw.Material(eps_r=_EPS_AVG), grid)

    assert str(aniso["mode_solver_kind"]).startswith("vector")

    n_aniso = float(aniso["effective_index"])
    n_iso_xx = float(iso_xx["effective_index"])
    n_iso_avg = float(iso_avg["effective_index"])

    # Guided fundamental between cladding (1) and the eps_xx core index.
    assert 1.0 < n_aniso < np.sqrt(_EPS_XX)

    # The anisotropic Ex mode must match the isotropic-eps_xx mode (identical
    # discretization: the TE eigenvalue depends only on eps_xx). This is the
    # core acceptance criterion for the anisotropic-aperture item.
    assert abs(n_aniso - n_iso_xx) < 5.0e-3 * n_iso_xx

    # It must be clearly separated from the isotropic-average mode: averaging the
    # diagonal (the pre-fix behavior) would have collapsed onto n_iso_avg.
    assert n_iso_avg < np.sqrt(_EPS_AVG)
    assert n_aniso - n_iso_avg > 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_scalar_complex_anisotropic_aperture_uses_polarization_component():
    """The lossy (scalar-complex) solve resolves eps_xx for the Ex mode."""
    grid = _slab_grid()
    f = grid["f"]

    im_eps = 0.1
    sigma_e = im_eps * (2.0 * np.pi * f) * VACUUM_PERMITTIVITY
    core = mw.Material(
        epsilon_tensor=mw.DiagonalTensor3(_EPS_XX, _EPS_YY, _EPS_ZZ), sigma_e=sigma_e, name="core"
    )

    compiled = _mode_effective_index(core, grid)

    # Loss routes the anisotropic plane to the complex scalar eigen-solve, which
    # sees the eps_xx polarization component.
    assert str(compiled["mode_solver_kind"]).startswith("scalar_complex")
    n_sim = compiled["effective_index_complex"]
    assert n_sim is not None

    solver = mw.Simulation.fdtd(_build_slab_scene(core, grid), frequencies=[f]).prepare().solver
    k0 = 2.0 * np.pi * f / float(solver.c)

    eps_xx = complex(_EPS_XX, -im_eps)
    n_ana = _analytic_even_te_neff(eps_xx, 1.0, k0, grid["a"])
    # The averaged-eps analytic reference the pre-fix solve would have produced.
    n_ana_avg = _analytic_even_te_neff(complex(_EPS_AVG, -im_eps), 1.0, k0, grid["a"])

    assert n_sim.imag < -1.0e-2
    assert abs(n_sim - n_ana) < 1.0e-3
    # The eps_xx and averaged references are well separated, so matching eps_xx
    # to 1e-3 unambiguously proves the polarization component (not the average).
    assert abs(n_ana.real - n_ana_avg.real) > 0.2
