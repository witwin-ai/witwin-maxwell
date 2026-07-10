"""FDTD static magnetic conductivity (sigma_m) coverage.

``sigma_m`` is the magnetic dual of ``sigma_e``: a magnetic conduction current
``sigma_m * H`` on Faraday's law, folded semi-implicitly into the H-update
decay/curl coefficients exactly as ``sigma_e`` folds into the E-update Ca/Cb.
These tests pin:
  1. the material constructs, is magnetic, and PEC rejects it,
  2. the semi-implicit Da/Db fold lands in ``chx_decay`` / ``chx_curl`` exactly,
  3. a magnetically-lossy slab attenuates a plane wave at the analytic rate
     ``alpha = (omega/c) Im(sqrt(1 + i sigma_m/(omega mu0)))`` (within 2%),
  4. a matched lossy layer (``sigma_m/mu0 = sigma_e/eps0``) reflects below a
     same-thickness PML baseline, while an unmatched (electric-only) layer of
     the same thickness reflects more,
  5. the FDTD adjoint still differentiates an eps design alongside a sigma_m
     structure (sigma_m folds into eps-independent H coefficients the reverse
     replay already consumes).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")

_EPS0 = 8.8541878128e-12
_MU0 = 4.0e-7 * np.pi
_C0 = 1.0 / np.sqrt(_EPS0 * _MU0)


def _to_np(tensor) -> np.ndarray:
    return np.asarray(torch.as_tensor(tensor).detach().cpu()).astype(float)


# ---------------------------------------------------------------------------
# 1. Material-level contract (no CUDA).
# ---------------------------------------------------------------------------
def test_sigma_m_material_is_magnetic():
    material = mw.Material(eps_r=1.0, sigma_m=1234.5)
    assert float(material.sigma_m) == pytest.approx(1234.5)
    assert material.capabilities().magnetic is True
    assert mw.Material(eps_r=2.0).capabilities().magnetic is False


def test_pec_material_rejects_sigma_m():
    with pytest.raises(ValueError, match="PEC Material"):
        mw.Material(pec=True, sigma_m=1.0)


# ---------------------------------------------------------------------------
# 2. Semi-implicit Da/Db fold lands exactly in the H-update coefficients.
# ---------------------------------------------------------------------------
def test_sigma_m_semi_implicit_coefficients_match_analytic():
    f = 1e9
    omega = 2.0 * np.pi * f
    sigma_m = 0.4 * omega * _MU0  # s = sigma_m / (omega mu0) = 0.4

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.09, 0.09), (-0.09, 0.09))),
        grid=mw.GridSpec.uniform(0.006),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1e7),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="magloss",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.12, 1.0, 1.0)),
            material=mw.Material(eps_r=1.0, mu_r=1.0, sigma_m=sigma_m),
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 1.0, 0.0),
            source_time=mw.CW(frequency=f, amplitude=1.0),
            name="pw",
        )
    )
    solver = mw.Simulation.fdtd(scene, frequencies=[f]).prepare().solver
    dt = float(solver.dt)

    half = 0.5 * sigma_m * dt / _MU0  # mu_r = 1 -> absolute mu = mu0 inside the slab
    decay_expected = (1.0 - half) / (1.0 + half)
    curl_expected = (dt / _MU0) / (1.0 + half)

    sigma_m_hx = solver.sigma_m_Hx
    chx_decay = solver.chx_decay
    chx_curl = solver.chx_curl
    inside = torch.isclose(sigma_m_hx, torch.full_like(sigma_m_hx, sigma_m), rtol=1e-3)
    assert bool(inside.any())
    # Away from the PML the fold is exactly the analytic semi-implicit Da/Db.
    assert torch.allclose(chx_decay[inside], torch.full_like(chx_decay[inside], decay_expected), rtol=2e-4)
    assert torch.allclose(chx_curl[inside], torch.full_like(chx_curl[inside], curl_expected), rtol=2e-4)

    # A magnetically-transparent (sigma_m == 0), PML-free cell keeps decay = 1,
    # curl = dt/mu0: the fold reduces to the lossless leapfrog exactly.
    vacuum = sigma_m_hx == 0
    assert torch.allclose(chx_decay[vacuum], torch.ones_like(chx_decay[vacuum]))
    assert torch.allclose(chx_curl[vacuum], torch.full_like(chx_curl[vacuum], dt / _MU0), rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. Analytic attenuation of a magnetically-lossy medium (acceptance a).
# ---------------------------------------------------------------------------
def test_sigma_m_attenuation_matches_analytic():
    f = 1e9
    omega = 2.0 * np.pi * f
    s = 0.4
    sigma_m = s * omega * _MU0
    # Magnetically-lossy relative permeability mu_r = 1 + i s (absorption), eps_r = 1.
    refractive_index = np.sqrt((1.0 + 1j * s) * 1.0)
    alpha_analytic = (omega / _C0) * float(np.imag(refractive_index))

    dx = 0.006
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.06, 0.06), (-0.06, 0.06))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1e7),
        device="cuda",
    )
    # Fill the whole domain: no front interface, so the forward wave on the +x
    # side of the source is a pure attenuating plane wave (ratio measurement is
    # independent of source calibration and any boundary reflection, which is
    # also forward-decaying at the same alpha).
    scene.add_structure(
        mw.Structure(
            name="fill",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(4.0, 4.0, 4.0)),
            material=mw.Material(eps_r=1.0, sigma_m=sigma_m),
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 1.0, 0.0),
            source_time=mw.CW(frequency=f, amplitude=1.0),
            name="pw",
        )
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[f],
        run_time=mw.TimeConfig.auto(steady_cycles=25, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()

    ey = np.abs(_to_np(result.E.y.detach().to(torch.complex64).abs()))
    nx, nyc, nzc = ey.shape
    profile = ey[:, nyc // 2, nzc // 2]
    xs = -0.6 + (np.arange(nx) + 0.5) * dx
    window = (xs > 0.0) & (xs < 0.42)  # interior, forward of the source, clear of +x PML
    slope = np.polyfit(xs[window], np.log(profile[window]), 1)[0]
    alpha_measured = -float(slope)

    rel_err = abs(alpha_measured - alpha_analytic) / alpha_analytic
    assert rel_err < 0.02, (
        f"sigma_m attenuation mismatch: measured alpha={alpha_measured:.4f} "
        f"analytic alpha={alpha_analytic:.4f} rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Matched lossy layer beats a same-thickness PML baseline (acceptance b).
# ---------------------------------------------------------------------------
_MATCH_F = 1e9
_MATCH_DX = 0.02
_MATCH_N = 10                       # absorber thickness in cells
_MATCH_T = _MATCH_N * _MATCH_DX     # 0.2 m
_MATCH_X_HIGH = 0.8
_MATCH_X_MON = -0.5
_MATCH_INC = (150, 420)             # incident-pulse time window at the monitor
_MATCH_REF = (600, 980)             # reflected-pulse time window (round trip ~570 steps)


def _reflection_coefficient(kind: str, *, alpha_t: float = 3.0, steps: int = 1200) -> float:
    """Peak reflected / peak incident |Ey| at a monitor in front of the +x absorber.

    ``kind`` selects the +x termination, all with the same 10-cell thickness:
    ``pml`` a CPML boundary, ``matched`` a PEC-backed matched lossy layer, and
    ``electric`` a PEC-backed electric-only (impedance-mismatched) lossy layer.
    """
    alpha = alpha_t / _MATCH_T
    sigma_e = alpha * _EPS0 * _C0
    sigma_m = sigma_e * _MU0 / _EPS0  # impedance-match condition sigma_m/mu0 = sigma_e/eps0
    if kind == "pml":
        boundary = mw.BoundarySpec.faces(
            default="none", num_layers=_MATCH_N, strength=1.0,
            x=("pml", "pml"), y=("pml", "pml"), z=("pml", "pml"),
        )
    else:
        boundary = mw.BoundarySpec.faces(
            default="none", num_layers=_MATCH_N, strength=1.0,
            x=("pml", "pec"), y=("pml", "pml"), z=("pml", "pml"),
        )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.2, _MATCH_X_HIGH), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(_MATCH_DX),
        boundary=boundary,
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 1.0, 0.0),
                source_time=mw.GaussianPulse(frequency=_MATCH_F, fwidth=0.4e9, amplitude=60.0),
                name="pw",
            )
        ],
    )
    if kind in ("matched", "electric"):
        material = (
            mw.Material(eps_r=1.0, sigma_e=sigma_e, sigma_m=sigma_m)
            if kind == "matched"
            else mw.Material(eps_r=1.0, sigma_e=sigma_e)
        )
        scene.add_structure(
            mw.Structure(
                name="dut",
                geometry=mw.Box(position=(_MATCH_X_HIGH - 0.5 * _MATCH_T, 0.0, 0.0), size=(_MATCH_T, 4.0, 4.0)),
                material=material,
            )
        )
    scene.add_monitor(mw.FieldTimeMonitor("probe", components=("Ey",), position=(_MATCH_X_MON, 0.0, 0.0), interval=1))
    result = mw.Simulation.fdtd(scene, frequencies=[_MATCH_F], run_time=mw.TimeConfig(time_steps=steps)).run()
    trace = _to_np(result.monitor("probe")["field"]).ravel()
    incident = np.max(np.abs(trace[_MATCH_INC[0]:_MATCH_INC[1]]))
    reflected = np.max(np.abs(trace[_MATCH_REF[0]:_MATCH_REF[1]]))
    return reflected / incident


def test_matched_lossy_layer_reflects_below_pml_baseline():
    reflection_pml = _reflection_coefficient("pml")
    reflection_matched = _reflection_coefficient("matched")
    reflection_electric = _reflection_coefficient("electric")

    # Acceptance (b): the matched layer reflects below the same-thickness PML.
    assert reflection_matched < reflection_pml, (
        f"matched reflection {reflection_matched:.3e} is not below the PML baseline "
        f"{reflection_pml:.3e}"
    )
    # The only difference between matched and electric is sigma_m, so the drop is
    # the impedance match at work: it must cut the reflection several-fold and put
    # the unmatched electric-only layer above the PML baseline.
    assert reflection_matched < 0.3 * reflection_electric, (
        f"matched reflection {reflection_matched:.3e} did not fall well below the "
        f"electric-only layer {reflection_electric:.3e}"
    )
    assert reflection_electric > reflection_pml


# ---------------------------------------------------------------------------
# 5. FDTD adjoint differentiates an eps design alongside a sigma_m structure.
# ---------------------------------------------------------------------------
class _MagneticLossDesignScene(mw.SceneModule):
    """Trainable eps design region next to a static magnetically-lossy structure."""

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
                name="magloss",
                geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                material=mw.Material(eps_r=1.0, sigma_m=5.0e3),
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


def test_fdtd_adjoint_accepts_magnetic_conductive_media():
    model = _MagneticLossDesignScene(init=0.0).cuda()
    simulation = mw.Simulation.fdtd(
        model,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=24),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )
    result = simulation.run()
    data = result.monitor("probe")["data"]
    loss = (data * data.conj()).real if data.is_complex() else data * data
    loss.backward()
    grad = model.logits.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.any(grad != 0)
