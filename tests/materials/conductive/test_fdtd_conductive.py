"""FDTD static electric conductivity (sigma_e) coverage.

These tests exercise the folded lossy-dielectric coefficient path added to the
FDTD update:
  1. a conductive material now runs without raising and produces finite fields,
  2. a conductive slab transmits with the same attenuation as the FDFD solver
     (primary correctness cross-check that catches a wrong Ca/Cb), and
  3. the FDTD adjoint bridge still rejects sigma_e media on the trainable path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")

# Shared conductive-slab geometry constants (used by the cross-solver check).
_SLAB_HALF = 0.48
_SLAB_RES = 0.03
_SLAB_FRONT_X = -0.15
_SLAB_BEHIND_X = 0.15


def _abs_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(dtype=torch.complex64).abs().cpu().numpy()


def test_conductive_material_runs_and_is_finite():
    """A nonzero sigma_e material must run an FDTD simulation without raising."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.03),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1e6),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="lossy_cube",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.12, 0.12, 0.12)),
            material=mw.Material(eps_r=2.0, sigma_e=0.05),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.09, 0.0, 0.0),
            width=0.04,
            polarization=(0.0, 0.0, 1.0),
            source_time=mw.CW(frequency=1e9, amplitude=50.0),
        )
    )

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=80),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
    ).run()

    assert torch.isfinite(result.E.z).all()
    assert torch.isfinite(result.E.x).all()
    assert torch.isfinite(result.E.y).all()


def _build_conductive_slab_scene():
    """A z-polarized point dipole in front of a full-transverse conductive slab."""
    half = _SLAB_HALF
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(_SLAB_RES),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1e7),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="conductive_slab",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.12, 1.2, 1.2)),
            material=mw.Material(eps_r=2.0, sigma_e=0.05),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.24, 0.0, 0.0),
            width=0.04,
            polarization=(0.0, 0.0, 1.0),
            source_time=mw.CW(frequency=1e9, amplitude=100.0),
        )
    )
    return scene


def _slab_transmission_ratio(ez_abs: np.ndarray) -> float:
    """Mean |Ez| behind the slab divided by mean |Ez| in front of the slab.

    Averaging over an interior transverse window and taking a front/behind ratio
    cancels the per-solver source-amplitude normalization and the shared geometric
    spreading, isolating the slab attenuation governed by the Ca/Cb coefficients.
    """
    nx = ez_abs.shape[0]
    front_i = min(int(round((_SLAB_FRONT_X + _SLAB_HALF) / _SLAB_RES)), nx - 1)
    behind_i = min(int(round((_SLAB_BEHIND_X + _SLAB_HALF) / _SLAB_RES)), nx - 1)
    lo, hi = 8, ez_abs.shape[1] - 8  # exclude PML-adjacent transverse cells
    hi = min(hi, ez_abs.shape[2] - 8)
    front = float(ez_abs[front_i, lo:hi, lo:hi].mean())
    behind = float(ez_abs[behind_i, lo:hi, lo:hi].mean())
    return behind / max(front, 1e-30)


def test_fdtd_matches_fdfd_conductive_slab_transmission():
    """FDTD and FDFD must agree on the conductive-slab transmission ratio."""
    freq = 1e9

    scene_fdfd = _build_conductive_slab_scene()
    scene_fdtd = _build_conductive_slab_scene()

    result_fdfd = mw.Simulation.fdfd(
        scene_fdfd,
        frequency=freq,
        solver=mw.GMRES(
            solver_type="sqmr",
            preconditioner="ssor",
            precision="double",
            max_iter=8000,
            tol=1e-7,
            restart=400,
        ),
        enable_plot=False,
        verbose=False,
    ).run()
    assert result_fdfd.solver.converged, "FDFD reference solve did not converge"
    result_fdtd = mw.Simulation.fdtd(
        scene_fdtd,
        frequencies=[freq],
        run_time=mw.TimeConfig.auto(steady_cycles=20, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()

    ez_fdfd = _abs_np(result_fdfd.E.z)
    ez_fdtd = _abs_np(result_fdtd.E.z)

    t_fdfd = _slab_transmission_ratio(ez_fdfd)
    t_fdtd = _slab_transmission_ratio(ez_fdtd)

    # The slab must actually be lossy in FDTD (folded coefficients took effect).
    assert t_fdtd < 0.8, f"FDTD slab not attenuating (T={t_fdtd:.3f}); Ca/Cb folding failed"
    assert t_fdfd < 0.8, f"FDFD slab not attenuating (T={t_fdfd:.3f})"

    rel_diff = abs(t_fdtd - t_fdfd) / max(t_fdtd, t_fdfd, 1e-30)
    assert rel_diff < 0.2, (
        f"FDTD vs FDFD conductive transmission mismatch: "
        f"T_fdtd={t_fdtd:.4f} T_fdfd={t_fdfd:.4f} rel_diff={rel_diff:.3f}"
    )


class _ConductiveDesignScene(mw.SceneModule):
    """Trainable eps design region alongside a static conductive structure."""

    def __init__(self, init: float = 0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                name="lossy",
                geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                material=mw.Material(eps_r=2.0, sigma_e=0.1),
            )
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, -0.06), size=(0.12, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.12),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.06), fields=("Ez",)))
        return scene


def test_fdtd_adjoint_rejects_conductive_media():
    """The trainable FDTD path must reject sigma_e media with a sigma_e message."""
    model = _ConductiveDesignScene(init=0.0).cuda()
    simulation = mw.Simulation.fdtd(
        model,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=24),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )
    with pytest.raises(NotImplementedError, match="sigma_e"):
        simulation.run()
