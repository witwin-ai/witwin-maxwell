from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
import witwin.maxwell.fdtd.excitation.modes as mode_solver
from witwin.maxwell.compiler.sources import _compile_mode_source
from witwin.maxwell.fdtd.excitation.modes import solve_mode_source_profile
from witwin.maxwell.scene import prepare_scene


_FREQUENCY = 2.0e9
_C0 = 299792458.0
_CORE_SIZE = (0.30, 0.15)
_APERTURE_SIZE = (0.525, 0.375)


def _rectangular_scene(
    spacing: float,
    *,
    mode_index: int,
    polarization: str = "Ez",
    core_size: tuple[float, float] = _CORE_SIZE,
    aperture_size: tuple[float, float] = _APERTURE_SIZE,
) -> mw.Scene:
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64),) * 3),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized"),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(2.56, core_size[0], core_size[1])),
            material=mw.Material(eps_r=4.0),
            name="guide",
        )
    )
    scene.add_source(
        mw.ModeSource(
            position=(-0.30, 0.0, 0.0),
            size=(0.0, aperture_size[0], aperture_size[1]),
            mode_index=mode_index,
            polarization=polarization,
            source_time=mw.CW(frequency=_FREQUENCY),
            name="mode",
        )
    )
    return scene


def _solve(scene: mw.Scene):
    prepared = prepare_scene(scene)
    context = SimpleNamespace(
        scene=prepared,
        Ex=torch.empty((1,), device=prepared.device, dtype=torch.float32),
        c=_C0,
        _compiled_material_model=prepared.compile_materials(),
    )
    compiled = _compile_mode_source(scene.sources[0], default_frequency=_FREQUENCY)
    return solve_mode_source_profile(context, compiled)


def _signed_node_count(profile: torch.Tensor, *, axis: int) -> int:
    values = profile.detach().cpu().numpy()
    line = values[:, values.shape[1] // 2] if axis == 0 else values[values.shape[0] // 2, :]
    significant = line[np.abs(line) > 0.1 * np.max(np.abs(line))]
    return int(np.sum(significant[:-1] * significant[1:] < 0.0))


def _normalized_overlap(left, right) -> float:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    return float(abs(np.vdot(left, right)) / (np.linalg.norm(left) * np.linalg.norm(right)))


def test_analytic_rectangular_waveguide_neff_and_higher_order_field_shape():
    result = _solve(_rectangular_scene(0.025, mode_index=1))

    k0 = 2.0 * math.pi * _FREQUENCY / _C0
    analytic_neff = math.sqrt(
        4.0
        - (2.0 * math.pi / (_CORE_SIZE[0] * k0)) ** 2
        - (math.pi / (_CORE_SIZE[1] * k0)) ** 2
    )
    assert abs(result["effective_index"] - analytic_neff) / analytic_neff < 0.01

    coords_u = result["coords_u"][1:-1].cpu().numpy()
    coords_v = result["coords_v"][1:-1].cpu().numpy()
    grid_u, grid_v = np.meshgrid(coords_u, coords_v, indexing="ij")
    core = (np.abs(grid_u) <= 0.5 * _CORE_SIZE[0] + 1e-12) & (
        np.abs(grid_v) <= 0.5 * _CORE_SIZE[1] + 1e-12
    )
    analytic_profile = np.zeros_like(grid_u)
    analytic_profile[core] = np.sin(
        2.0 * math.pi * (grid_u[core] + 0.5 * _CORE_SIZE[0]) / _CORE_SIZE[0]
    ) * np.sin(math.pi * (grid_v[core] + 0.5 * _CORE_SIZE[1]) / _CORE_SIZE[1])
    assert _normalized_overlap(result["profile"], analytic_profile) > 0.95
    assert _signed_node_count(result["profile"], axis=0) == 1
    assert _signed_node_count(result["profile"], axis=1) == 0


def test_non_degenerate_mode_ordering_and_mode_index_are_physical():
    fundamental = _solve(_rectangular_scene(0.025, mode_index=0))
    higher_order = _solve(_rectangular_scene(0.025, mode_index=1))

    assert fundamental["effective_index"] > higher_order["effective_index"]
    assert _signed_node_count(fundamental["profile"], axis=0) == 0
    assert _signed_node_count(higher_order["profile"], axis=0) == 1
    selected = higher_order["candidate_diagnostics"][higher_order["selected_candidate_index"]]
    assert selected["family_index"] == 1
    assert selected["status"] == "eligible"


def test_higher_order_mode_converges_on_three_grids_without_index_changes():
    results = [_solve(_rectangular_scene(spacing, mode_index=1)) for spacing in (0.0375, 0.025, 0.0125)]
    effective_indices = [result["effective_index"] for result in results]

    assert all(_signed_node_count(result["profile"], axis=0) == 1 for result in results)
    assert all(
        result["candidate_diagnostics"][result["selected_candidate_index"]]["family_index"] == 1
        for result in results
    )
    assert (max(effective_indices) - min(effective_indices)) / effective_indices[-1] < 0.01


def test_square_degenerate_subspace_rotates_to_stable_requested_polarizations():
    square = (0.30, 0.30)
    aperture = (0.525, 0.525)
    ez = _solve(
        _rectangular_scene(
            0.025,
            mode_index=0,
            polarization="Ez",
            core_size=square,
            aperture_size=aperture,
        )
    )
    ey = _solve(
        _rectangular_scene(
            0.025,
            mode_index=0,
            polarization="Ey",
            core_size=square,
            aperture_size=aperture,
        )
    )

    ez_selected = ez["candidate_diagnostics"][ez["selected_candidate_index"]]
    ey_selected = ey["candidate_diagnostics"][ey["selected_candidate_index"]]
    assert ez["effective_index"] == pytest.approx(ey["effective_index"], rel=1e-6)
    assert ez_selected["polarization_fraction"] > 0.99
    assert ey_selected["polarization_fraction"] > 0.99
    assert len(ez_selected["raw_indices"]) == 2
    assert _normalized_overlap(
        ez["component_profiles"]["Ez"],
        ey["component_profiles"]["Ey"].T,
    ) > 0.99


def test_slightly_split_degenerate_pair_still_rotates_to_requested_polarization():
    interior_u = interior_v = 3
    block = interior_u * interior_v
    profile = np.ones((interior_u, interior_v), dtype=np.float64)
    zeros = np.zeros_like(profile)
    preferred = np.concatenate((-profile.reshape(block), zeros.reshape(block), zeros.reshape(block), profile.reshape(block)))
    orthogonal = np.concatenate((zeros.reshape(block), profile.reshape(block), profile.reshape(block), zeros.reshape(block)))
    mixed = np.stack(
        (
            (preferred + orthogonal) / math.sqrt(2.0),
            (preferred - orthogonal) / math.sqrt(2.0),
        ),
        axis=1,
    )

    _, _, profiles, diagnostics = mode_solver._select_and_normalize_vector_mode_numpy(
        np.array([5.0, 5.0 * (1.0 - 2.0e-5)]),
        mixed,
        interior_u=interior_u,
        interior_v=interior_v,
        mode_index=0,
        field_names=("Eu", "Ev", "Hu", "Hv"),
        preferred_field_name="Ev",
    )

    selected = diagnostics["candidates"][diagnostics["selected_candidate_index"]]
    assert len(selected["raw_indices"]) == 2
    assert selected["polarization_fraction"] > 0.99
    assert _normalized_overlap(profiles["Ev"], profile) > 0.99


def test_dense_and_sparse_higher_order_candidates_have_matching_order(monkeypatch):
    scene = _rectangular_scene(0.0375, mode_index=1)
    dense = _solve(scene)
    monkeypatch.setattr(mode_solver, "_FULL_VECTOR_DENSE_LIMIT", 0)
    sparse = _solve(scene)

    assert dense["mode_solver_kind"] == "vector_dense"
    assert sparse["mode_solver_kind"] == "vector_sparse"
    assert sparse["effective_index"] == pytest.approx(dense["effective_index"], rel=1e-5)
    assert _normalized_overlap(sparse["profile"], dense["profile"]) > 0.999
    dense_families = [
        entry["family_index"] for entry in dense["candidate_diagnostics"] if entry["family_index"] is not None
    ]
    sparse_families = [
        entry["family_index"] for entry in sparse["candidate_diagnostics"] if entry["family_index"] is not None
    ]
    assert sparse_families[:3] == dense_families[:3]


def test_numpy_and_torch_candidate_sorting_are_identical():
    eigenvalues = np.array([3.0 + 2.0e-8j, 3.0 + 1.0e-8j, 2.0 + 0.0j, -1.0 + 0.0j])
    eigenvectors = np.eye(4, dtype=np.complex128)
    beta_numpy, vector_numpy = mode_solver._select_vector_mode_numpy(
        eigenvalues,
        eigenvectors,
        interior_u=1,
        interior_v=1,
        mode_index=0,
    )
    beta_torch, vector_torch = mode_solver._select_vector_mode_torch(
        torch.as_tensor(eigenvalues),
        torch.as_tensor(eigenvectors),
        mode_index=0,
    )

    assert complex(beta_torch.item()) == pytest.approx(complex(beta_numpy))
    assert np.allclose(vector_torch.numpy(), vector_numpy)


def test_duplicate_and_checkerboard_candidates_do_not_consume_mode_index():
    interior_u = interior_v = 5
    block = interior_u * interior_v
    coords = np.arange(1, interior_u + 1, dtype=np.float64)
    smooth = np.sin(math.pi * coords / (interior_u + 1))[:, None]
    smooth = smooth * np.sin(math.pi * coords / (interior_v + 1))[None, :]
    checkerboard = (-1.0) ** (np.add.outer(np.arange(interior_u), np.arange(interior_v)))

    def vector(profile):
        zeros = np.zeros_like(profile)
        return np.concatenate((-profile.reshape(block), zeros.reshape(block), zeros.reshape(block), profile.reshape(block)))

    eigenvalues = np.array([6.0, 5.0, 5.0 * (1.0 - 1.0e-8)])
    eigenvectors = np.stack((vector(checkerboard), vector(smooth), vector(smooth)), axis=1)
    beta, _, profiles, diagnostics = mode_solver._select_and_normalize_vector_mode_numpy(
        eigenvalues,
        eigenvectors,
        interior_u=interior_u,
        interior_v=interior_v,
        mode_index=0,
        field_names=("Eu", "Ev", "Hu", "Hv"),
        preferred_field_name="Ev",
    )

    assert beta == pytest.approx(5.0)
    assert _normalized_overlap(profiles["Ev"], smooth) > 0.999
    assert diagnostics["raw_candidate_count"] == 3
    assert diagnostics["independent_candidate_count"] == 2
    assert diagnostics["candidates"][0]["status"] == "checkerboard"
    assert diagnostics["candidates"][0]["checkerboard_fraction"] > 0.5


def test_candidate_power_gram_and_discrete_divergence_are_orthogonal():
    result = _solve(_rectangular_scene(0.025, mode_index=1))
    matrix = np.asarray(result["candidate_overlap_matrix"])
    off_diagonal = matrix - np.eye(matrix.shape[0])
    selected = result["candidate_diagnostics"][result["selected_candidate_index"]]

    assert np.max(np.abs(off_diagonal)) < 1e-6
    assert selected["eigenpair_residual"] < 1e-10
    assert selected["electric_divergence_residual"] < 1e-10
    assert selected["magnetic_divergence_residual"] < 1e-10
    assert selected["checkerboard_fraction"] < mode_solver._VECTOR_CHECKERBOARD_FRACTION_LIMIT
