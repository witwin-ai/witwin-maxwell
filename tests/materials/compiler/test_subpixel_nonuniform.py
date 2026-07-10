"""Subpixel material averaging on nonuniform (custom / auto) Yee grids.

Before P5.3 the subpixel sample offsets were scalar multiples of ``Scene.dx``,
so ``samples > 1`` raised on any ``GridSpec.custom`` / ``GridSpec.auto`` grid
(the scalar spacing is undefined there). The offsets are now per-node fields
scaled by the local Yee dual-cell width, which reduces bit-exactly to the scalar
spacing on a uniform grid and generalizes both the arithmetic and the polarized
(Kottke) averaging to graded grids.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _axis_averaging_width, _axis_sample_offsets
from witwin.maxwell.scene import prepare_scene


# --- Regression guard: uniform grid stays bit-close through the new per-node path ---

def _rich_scene(grid, bounds, *, samples, averaging):
    """Scene exercising every accumulated material channel at once.

    Three disjoint boxes cover isotropic eps + conductivity + instantaneous
    nonlinearity (chi2/chi3/TPA), space-time modulation, and a dispersive
    Lorentz pole, so the regression guard compares the full compiled model and
    not just the linear permittivity.
    """
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds),
        grid=grid,
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=samples, averaging=averaging),
    )
    scene.add_structure(
        mw.Structure(
            name="nl",
            geometry=mw.Box(position=(-0.12, 0.05, 0.0), size=(0.3, 0.3, 0.3)),
            material=mw.Material(
                eps_r=6.0,
                sigma_e=0.3,
                kerr_chi3=1e-18,
                nonlinearity=(
                    mw.NonlinearSusceptibility(chi2=2e-12),
                    mw.TwoPhotonAbsorption(beta=1e-11),
                ),
            ),
        )
    )
    scene.add_structure(
        mw.Structure(
            name="mod",
            geometry=mw.Box(position=(0.14, -0.04, 0.02), size=(0.28, 0.28, 0.28)),
            material=mw.Material(
                eps_r=4.0,
                modulation=mw.ModulationSpec(frequency=1e9, amplitude=0.1, phase=0.4),
            ),
        )
    )
    scene.add_structure(
        mw.Structure(
            name="disp",
            geometry=mw.Box(position=(0.0, 0.2, -0.15), size=(0.25, 0.25, 0.25)),
            material=mw.Material(
                eps_r=2.0,
                lorentz_poles=(
                    mw.LorentzPole(delta_eps=1.5, resonance_frequency=2e9, gamma=1e8),
                ),
            ),
        )
    )
    return prepare_scene(scene)


def _flatten_model(model):
    flat = {}
    for key in ("eps_components", "mu_components", "sigma_e_components", "eps_offdiag_components"):
        for axis, tensor in model[key].items():
            flat[f"{key}.{axis}"] = tensor
    for key in ("kerr_chi3", "chi2", "tpa_sigma", "modulation_cos", "modulation_sin", "modulation_omega"):
        flat[key] = model[key]
    for family in ("debye_poles", "drude_poles", "lorentz_poles", "mu_debye_poles", "mu_drude_poles", "mu_lorentz_poles"):
        for index, entry in enumerate(model[family]):
            flat[f"{family}.{index}"] = entry["weight"]
    return flat


@pytest.mark.parametrize(
    "samples, averaging",
    [((2, 2, 2), "arithmetic"), ((3, 2, 3), "arithmetic"), ((2, 2, 2), "polarized")],
)
def test_uniform_grid_custom_masters_subpixel_bit_close(samples, averaging):
    # A uniform grid re-expressed as GridSpec.custom from its own node masters
    # takes the new per-node offset path; the compiled model must match the
    # scalar-spacing GridSpec.uniform run channel-for-channel, bitwise. This is
    # the P5.3 bit-close regression guard for every accumulated channel.
    dl = 0.05
    bounds = ((-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4))
    uniform = _rich_scene(mw.GridSpec.uniform(dl), bounds, samples=samples, averaging=averaging)
    uniform_model = uniform.compile_materials()

    custom_grid = mw.GridSpec.custom(uniform.x_nodes64, uniform.y_nodes64, uniform.z_nodes64)
    custom_bounds = tuple(
        (float(nodes[0]), float(nodes[-1]))
        for nodes in (uniform.x_nodes64, uniform.y_nodes64, uniform.z_nodes64)
    )
    custom_model = _rich_scene(custom_grid, custom_bounds, samples=samples, averaging=averaging).compile_materials()

    uniform_flat = _flatten_model(uniform_model)
    custom_flat = _flatten_model(custom_model)
    assert set(uniform_flat) == set(custom_flat)
    for name, tensor in uniform_flat.items():
        assert torch.equal(tensor, custom_flat[name]), name

    # The guard is only meaningful if the exotic channels are actually populated.
    for channel in ("kerr_chi3", "chi2", "tpa_sigma", "modulation_cos", "modulation_omega"):
        assert float(uniform_flat[channel].abs().max()) > 0.0, channel


# --- Centerpiece: GridSpec.auto + subpixel now compiles (arithmetic AND polarized) ---

@pytest.mark.parametrize("averaging", ["arithmetic", "polarized"])
def test_auto_grid_subpixel_compiles(averaging):
    # Pre-P5.3 this raised inside _sample_offsets via the scalar Scene.dx.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.auto(min_steps_per_wavelength=10, wavelength=0.3),
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=(2, 2, 2), averaging=averaging),
    )
    scene.add_structure(
        mw.Structure(
            name="sphere",
            geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.15),
            material=mw.Material(eps_r=12.0),
        )
    )
    prepared = prepare_scene(scene)
    assert prepared.grid.is_custom  # auto resolved to a nonuniform grid
    eps, mu = prepared.compile_material_components()
    for axis in ("x", "y", "z"):
        assert torch.isfinite(eps[axis]).all()
        assert torch.isfinite(mu[axis]).all()
        # Subpixel averaging keeps boundary cells strictly between vacuum and eps=12.
        assert float(eps[axis].min()) >= 1.0 - 1e-6
        assert float(eps[axis].max()) <= 12.0 + 1e-6
    # The interface is genuinely smoothed: some node lies strictly inside (1, 12).
    interior = (eps["x"] > 1.0 + 1e-3) & (eps["x"] < 12.0 - 1e-3)
    assert int(interior.sum()) > 0


# --- Per-node averaging window scales with the local cell ---

def test_subpixel_offsets_scale_with_local_dual_cell():
    graded = np.array(
        [-0.5, -0.30, -0.15, -0.05, 0.0, 0.08, 0.20, 0.35, 0.5], dtype=np.float64
    )
    scene = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
            grid=mw.GridSpec.custom(graded, graded, graded),
            device="cpu",
            subpixel_samples=mw.SubpixelSpec(samples=(2, 1, 1)),
        )
    )
    width = _axis_averaging_width(scene, "x")
    assert torch.is_tensor(width)
    # The window equals the per-node Yee dual-cell width, so it varies across the axis.
    expected = torch.as_tensor(scene.dx_dual64, dtype=torch.float32)
    assert torch.equal(width, expected)
    assert float(width.max()) > 3.0 * float(width.min())  # genuinely graded

    offsets = _axis_sample_offsets(scene, "x", 2)
    assert len(offsets) == 2
    assert torch.is_tensor(offsets[0]) and torch.is_tensor(offsets[1])
    torch.testing.assert_close(offsets[0], -0.25 * width, rtol=0.0, atol=0.0)
    torch.testing.assert_close(offsets[1], +0.25 * width, rtol=0.0, atol=0.0)

    # A uniform axis keeps the scalar (bit-exact) offset path.
    uniform = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            grid=mw.GridSpec.uniform(0.1),
            device="cpu",
            subpixel_samples=mw.SubpixelSpec(samples=(2, 1, 1)),
        )
    )
    scalar_offsets = _axis_sample_offsets(uniform, "x", 2)
    assert scalar_offsets == [-0.025, 0.025]


# --- Both averaging modes stay physically correct on a genuinely graded grid ---

def test_graded_grid_interface_arithmetic_and_polarized():
    # An interior node sits exactly at x=0 with unequal neighbor cells (0.05 vs
    # 0.08). A slab fills the +x half so its face lands on that node.
    xg = np.array([-0.5, -0.30, -0.15, -0.05, 0.0, 0.08, 0.20, 0.35, 0.5], dtype=np.float64)
    yg = np.linspace(-0.5, 0.5, 11)
    zg = np.linspace(-0.5, 0.5, 11)
    node = (int(np.argmin(np.abs(xg))), 5, 5)
    eps_r = 12.0
    arithmetic_mean = 0.5 * 1.0 + 0.5 * eps_r

    def _eps(averaging, samples):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
            grid=mw.GridSpec.custom(xg, yg, zg),
            device="cpu",
            subpixel_samples=mw.SubpixelSpec(samples=samples, averaging=averaging),
        )
        scene.add_structure(
            mw.Structure(
                name="slab",
                geometry=mw.Box(position=(0.25, 0.0, 0.0), size=(0.5, 2.0, 2.0)),
                material=mw.Material(eps_r=eps_r),
            )
        )
        eps, _ = prepare_scene(scene).compile_material_components()
        return eps

    # Arithmetic: the symmetric per-node window is half inside the slab on every
    # axis, so all three components equal the arithmetic mean.
    eps_arith = _eps("arithmetic", (2, 2, 2))
    for axis in ("x", "y", "z"):
        assert abs(eps_arith[axis][node].item() - arithmetic_mean) < 1e-2

    # Polarized: the normal (x) component leans toward the harmonic mean while
    # the tangential (y) component stays arithmetic.
    eps_pol = _eps("polarized", (6, 6, 6))
    eps_x = eps_pol["x"][node].item()
    eps_y = eps_pol["y"][node].item()
    assert abs(eps_y - arithmetic_mean) < 5e-2
    assert eps_x < eps_y - 0.1  # normal component pulled below arithmetic
    assert eps_x > 1.0 / (0.5 / 1.0 + 0.5 / eps_r)  # bounded below by harmonic mean


# --- End-to-end: auto + polarized subpixel runs through the FDTD solver ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_auto_grid_subpixel_fdtd_smoke():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.35, 0.35), (-0.35, 0.35), (-0.35, 0.35))),
        grid=mw.GridSpec.auto(),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
        subpixel_samples=mw.SubpixelSpec(samples=(2, 2, 2), averaging="polarized"),
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Sphere(position=(0.05, 0.0, 0.0), radius=0.1),
            material=mw.Material(eps_r=6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9),
            name="src",
        )
    )
    solver = mw.Simulation.fdtd(scene, frequency=1e9).prepare().solver
    solver.solve(time_steps=40, dft_frequency=None, dft_window="none", full_field_dft=False)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.isfinite(getattr(solver, name)).all(), name
    assert float(torch.abs(solver.Ez).max()) > 0.0
