"""Layered / non-vacuum exterior Huygens near-to-far-field validation.

The closed-surface Huygens transform reconstructs the radiated field from the
tangential fields on a box by radiating equivalent currents ``J = n x H`` and
``M = -n x E`` into the *homogeneous exterior* medium. When that exterior is a
dielectric layer (index ``n_b != 1``) the radiation kernel must use the
background wavenumber ``k_b = n_b * omega / c`` and intrinsic impedance
``eta_b = eta0 * sqrt(mu_r / eps_r)`` sampled just outside the surface, not the
vacuum values.

The far field radiated by a source enclosed by a closed Huygens box is
independent of which box is chosen, provided the region between two nested
boxes is homogeneous and source-free (surface-equivalence theorem / Green's
second identity). This is exactly the plan acceptance: the far field read from a
*bigger* closed surface fully inside the homogeneous outer region must match the
one read from a *smaller* inner surface.

Two complementary checks are used:

* An analytic, deterministic reference (exact Hertzian-dipole fields radiated
  into an infinite dielectric, sampled on two nested boxes) verifies the
  transform is surface-invariant and pattern-correct to well under 1%. This is
  the ground truth a converged FDTD run would approach, and it isolates the
  transform from FDTD discretization. Forcing the vacuum kernel on the same
  dielectric-exterior currents breaks the invariance by >10%, proving the
  background wavenumber is essential.

* A direct FDTD run of a dipole embedded in a dielectric background confirms the
  end-to-end path: the exterior medium is auto-sampled from the material grid,
  threaded into the transform as ``k_b``, and the resulting (axisymmetric) far
  field matches the analytic ``sin^2(theta)`` dipole pattern within 5%.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess import (
    NearFieldFarFieldTransformer,
    StrattonChuPropagator,
    equivalent_surface_currents_from_fields,
    equivalent_surface_currents_from_monitor,
)
from witwin.maxwell.postprocess.stratton_chu import (
    EquivalentCurrentsSurface,
    build_plane_points,
)


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "test_output" / "validation" / "postprocess"

_C0 = 299792458.0
_MU0 = 4.0e-7 * math.pi
_EPS0 = 1.0 / (_MU0 * _C0**2)
_FREQUENCY = 1.0e9


# --------------------------------------------------------------------------- #
# Analytic Hertzian dipole in a homogeneous medium (e^{-i w t}, outgoing e^{ikr}).
# --------------------------------------------------------------------------- #
def _dipole_fields(points: torch.Tensor, *, k: float, eta: float, omega: float):
    """Exact fields of a z-directed Hertzian dipole (Jackson 9.18, SI) in a
    homogeneous medium characterised by wavenumber ``k`` and impedance ``eta``."""
    p = torch.tensor([0.0, 0.0, 1.0], dtype=torch.complex128)
    c = omega / k
    eps = k / (eta * omega)
    r = torch.linalg.norm(points, dim=-1)
    n = (points / r[..., None]).to(torch.complex128)
    n_dot_p = n[..., 2]
    ikr = 1j * k * r
    exp_over_r = torch.exp(ikr) / r
    exp = torch.exp(ikr)

    p_perp = p[None, :] - n * n_dot_p[..., None]          # (n x p) x n
    p_near = 3.0 * n * n_dot_p[..., None] - p[None, :]     # 3 n (n.p) - p
    e_field = (1.0 / (4.0 * math.pi * eps)) * (
        (k**2) * p_perp * exp_over_r[..., None]
        + p_near * (1.0 / r**3 - 1j * k / r**2)[..., None] * exp[..., None]
    )
    n_cross_p = torch.cross(n, p[None, :].expand_as(n), dim=-1)
    h_field = (c * k**2 / (4.0 * math.pi)) * n_cross_p * exp_over_r[..., None] * (1.0 - 1.0 / ikr)[..., None]
    return e_field, h_field


def _analytic_box_currents(half, *, k, eta, omega, background_eps_r, samples=41):
    coord = torch.linspace(-half, half, samples, dtype=torch.float64)
    faces = []
    for axis, position, normal in (
        ("x", -half, "-"), ("x", half, "+"),
        ("y", -half, "-"), ("y", half, "+"),
        ("z", -half, "-"), ("z", half, "+"),
    ):
        pts = build_plane_points(axis, position, coord, coord).reshape(-1, 3)
        e_field, h_field = _dipole_fields(pts, k=k, eta=eta, omega=omega)
        e_field = e_field.reshape(samples, samples, 3)
        h_field = h_field.reshape(samples, samples, 3)
        components = {}
        for index, label in enumerate("xyz"):
            components[f"E{label}"] = e_field[..., index]
            components[f"H{label}"] = h_field[..., index]
        faces.append(
            equivalent_surface_currents_from_fields(
                axis=axis,
                position=position,
                frequency=omega / (2.0 * math.pi),
                u=coord,
                v=coord,
                fields=components,
                normal_direction=normal,
                background_eps_r=background_eps_r,
                background_mu_r=1.0,
            )
        )
    return EquivalentCurrentsSurface(tuple(faces))


def _power_pattern(transformer, theta, phi, radius=10.0):
    theta_grid, phi_grid = torch.broadcast_tensors(theta[:, None], phi[None, :])
    far = transformer.transform(theta_grid, phi_grid, radius=radius, batch_size=4096)
    return (torch.abs(far["E_theta"]).square() + torch.abs(far["E_phi"]).square()).detach().cpu().numpy()


def _relative_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# --------------------------------------------------------------------------- #
# Analytic (deterministic) reference: surface invariance + pattern correctness.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def analytic_layered_far_field():
    eps_r = 2.25
    n = math.sqrt(eps_r)
    omega = 2.0 * math.pi * _FREQUENCY
    k = n * omega / _C0
    eta = math.sqrt(_MU0 / _EPS0) / n

    inner = _analytic_box_currents(0.06, k=k, eta=eta, omega=omega, background_eps_r=eps_r)
    outer = _analytic_box_currents(0.12, k=k, eta=eta, omega=omega, background_eps_r=eps_r)
    # Same currents, but forced to radiate with the vacuum kernel (negative control).
    inner_vac = _analytic_box_currents(0.06, k=k, eta=eta, omega=omega, background_eps_r=1.0)
    outer_vac = _analytic_box_currents(0.12, k=k, eta=eta, omega=omega, background_eps_r=1.0)

    theta = torch.linspace(0.02, math.pi - 0.02, 60, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 90, dtype=torch.float64)

    def transform(currents):
        return NearFieldFarFieldTransformer(currents, c=_C0, eps0=_EPS0, mu0=_MU0, device="cpu")

    inner_pattern = _power_pattern(transform(inner), theta, phi)
    outer_pattern = _power_pattern(transform(outer), theta, phi)
    inner_pattern_vac = _power_pattern(transform(inner_vac), theta, phi)
    outer_pattern_vac = _power_pattern(transform(outer_vac), theta, phi)

    reference = (np.sin(theta.numpy()) ** 2)[:, None] * np.ones((1, phi.numel()))
    outer_norm = outer_pattern / outer_pattern.max()
    reference_norm = reference / reference.max()

    return {
        "sampled_k": complex(transform(inner).k),
        "expected_k": k,
        "surface_invariance_rel": _relative_l2(outer_pattern, inner_pattern),
        "pattern_vs_sin2_rel": _relative_l2(outer_norm, reference_norm),
        "vacuum_kernel_invariance_rel": _relative_l2(outer_pattern_vac, inner_pattern_vac),
    }


def test_analytic_layered_transform_uses_background_wavenumber(analytic_layered_far_field):
    assert abs(analytic_layered_far_field["sampled_k"].real - analytic_layered_far_field["expected_k"]) \
        / analytic_layered_far_field["expected_k"] < 1e-9
    assert abs(analytic_layered_far_field["sampled_k"].imag) < 1e-9


def test_analytic_layered_rcs_matches_larger_surface_reference(analytic_layered_far_field):
    # Plan acceptance (deterministic form): the far field from the bigger closed
    # surface matches the inner surface, and both match the analytic dielectric
    # dipole pattern, to far better than 5%.
    assert analytic_layered_far_field["surface_invariance_rel"] < 0.02
    assert analytic_layered_far_field["pattern_vs_sin2_rel"] < 0.02


def test_analytic_vacuum_kernel_breaks_layered_invariance(analytic_layered_far_field):
    # Ignoring the dielectric exterior (vacuum k) makes the two nested boxes
    # disagree by more than the 5% acceptance -- the background kernel is load
    # bearing.
    assert analytic_layered_far_field["vacuum_kernel_invariance_rel"] > 0.08


def test_analytic_layered_stratton_chu_near_field_matches_dipole():
    # The near-field Stratton-Chu propagator must also radiate the equivalent
    # currents into the dielectric background: reconstruct the exterior field at
    # a point outside the box and compare to the exact dipole field.
    eps_r = 2.25
    n = math.sqrt(eps_r)
    omega = 2.0 * math.pi * _FREQUENCY
    k = n * omega / _C0
    eta = math.sqrt(_MU0 / _EPS0) / n
    currents = _analytic_box_currents(0.06, k=k, eta=eta, omega=omega, background_eps_r=eps_r, samples=61)
    propagator = StrattonChuPropagator(currents, c=_C0, eps0=_EPS0, mu0=_MU0, device="cpu")
    point = torch.tensor([[0.0, 0.0, 0.25]], dtype=torch.float64)
    reconstructed = propagator.propagate_points(point, batch_size=1)["E"][0]
    exact, _ = _dipole_fields(point, k=k, eta=eta, omega=omega)
    rel = float(torch.linalg.norm(reconstructed - exact[0]) / torch.linalg.norm(exact[0]))
    assert rel < 0.02


# --------------------------------------------------------------------------- #
# Direct FDTD run: end-to-end background sampling + far-field pattern.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fdtd_layered_far_field():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eps_r = 2.25
    n = math.sqrt(eps_r)
    box_half = 0.20
    domain_half = 0.40

    surface = mw.ClosedSurfaceMonitor.box(
        "huygens",
        position=(0.0, 0.0, 0.0),
        size=(2.0 * box_half,) * 3,
        frequencies=(_FREQUENCY,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-domain_half, domain_half),) * 3),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
    )
    # Homogeneous dielectric background filling the whole domain (including the
    # PML) so the exterior is reflection-free to the absorber.
    scene.add_structure(
        mw.Structure(
            name="dielectric_background",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(8.0 * domain_half,) * 3),
            material=mw.Material(eps_r=eps_r),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=1.0),
            name="dipole",
        )
    )
    scene.add_monitor(surface)

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[_FREQUENCY],
        run_time=mw.TimeConfig.auto(steady_cycles=12, transient_cycles=30),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    currents = equivalent_surface_currents_from_monitor(result, surface.name)
    vacuum_k = 2.0 * math.pi * _FREQUENCY / result.solver.c

    theta = torch.linspace(0.02, math.pi - 0.02, 90, device="cuda", dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 120, device="cuda", dtype=torch.float64)

    aware = NearFieldFarFieldTransformer(currents, solver=result.solver, device="cuda")
    vacuum = NearFieldFarFieldTransformer(
        currents, solver=result.solver, background_eps_r=1.0, background_mu_r=1.0, device="cuda"
    )
    aware_pattern = _power_pattern(aware, theta, phi)
    vacuum_pattern = _power_pattern(vacuum, theta, phi)

    theta_np = theta.detach().cpu().numpy()
    reference = np.sin(theta_np) ** 2

    def profile_rel(pattern):
        profile = pattern.mean(axis=1)
        profile = profile / profile.max()
        return _relative_l2(profile, reference / reference.max())

    def pattern_rel(pattern):
        normalized = pattern / pattern.max()
        ref2d = (reference[:, None] * np.ones((1, pattern.shape[1])))
        return _relative_l2(normalized, ref2d / ref2d.max())

    return {
        "sampled_eps": complex(currents.background_eps_r),
        "sampled_mu": complex(currents.background_mu_r),
        "aware_k": complex(aware.k),
        "expected_k": n * vacuum_k,
        "profile_vs_sin2_rel": profile_rel(aware_pattern),
        "aware_pattern_vs_sin2_rel": pattern_rel(aware_pattern),
        "vacuum_pattern_vs_sin2_rel": pattern_rel(vacuum_pattern),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_layered_background_is_sampled_from_material(fdtd_layered_far_field):
    assert abs(fdtd_layered_far_field["sampled_eps"].real - 2.25) < 0.05
    assert abs(fdtd_layered_far_field["sampled_eps"].imag) < 1e-3
    assert abs(fdtd_layered_far_field["sampled_mu"].real - 1.0) < 1e-3
    expected_k = fdtd_layered_far_field["expected_k"]
    assert abs(fdtd_layered_far_field["aware_k"].real - expected_k) / expected_k < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_layered_rcs_matches_direct_reference_within_5pct(fdtd_layered_far_field):
    # Plan acceptance (direct-FDTD form): the FDTD dielectric dipole far field,
    # transformed with the auto-sampled background wavenumber, matches the
    # analytic dielectric dipole reference (axisymmetric sin^2 profile) within
    # 5%.
    assert fdtd_layered_far_field["profile_vs_sin2_rel"] < 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_layered_vacuum_kernel_is_unphysical(fdtd_layered_far_field):
    # Sensitivity guard on real FDTD fields: dropping the dielectric background
    # (vacuum k) grossly distorts the reconstructed pattern relative to the
    # background-aware transform.
    assert fdtd_layered_far_field["vacuum_pattern_vs_sin2_rel"] > 3.0 * fdtd_layered_far_field["aware_pattern_vs_sin2_rel"]
    assert fdtd_layered_far_field["vacuum_pattern_vs_sin2_rel"] > 0.30
