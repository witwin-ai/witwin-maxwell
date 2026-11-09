"""Prepare-only source term counts and patches compared with baseline behavior."""

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD preparation requires CUDA")


def _scene(length, source):
    return mw.Scene(
        domain=mw.Domain(bounds=((-length, length),) * 3),
        grid=mw.GridSpec.uniform(.1 * length), boundary=mw.BoundarySpec.none(),
        sources=[source], device="cuda",
    )


def _assert_term_patches_equal(actual, baseline):
    assert len(actual) == len(baseline)
    for term, reference in zip(actual, baseline):
        assert term["field_name"] == reference["field_name"]
        assert term["offsets"] == reference["offsets"]
        for key in ("patch", "cw_cos_patch", "cw_sin_patch"):
            if reference[key] is None:
                assert term[key] is None
            else:
                assert torch.equal(term[key], reference[key])


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("case, expected", [("oblique_angles", (4, 8)), ("oblique_exact", (4, 8)),
                                          ("axis_angles", (4, 4)), ("axis_exact", (4, 4))])
def test_plane_wave_tfsf_term_counts(length, case, expected):
    phi = math.pi / 4 if case.startswith("oblique") else 0.
    residual = math.cos(math.pi / 2) if case.endswith("angles") else 0.
    direction = (math.cos(phi), math.sin(phi), residual)
    polarization = (-math.sin(phi), math.cos(phi), 0.)
    source = mw.PlaneWave(
        direction=direction, polarization=polarization,
        source_time=mw.CW(frequency=C_0 / (1.55 * length)),
        injection=mw.TFSF(bounds=((- .4 * length, .4 * length),) * 3), name="incident",
    )
    solver = mw.Simulation.fdtd(_scene(length, source), frequency=source.source_time.frequency).prepare().solver
    state = solver._tfsf_state
    # Each nonzero incident component contributes on four faces: H_z, and E_x/E_y
    # for oblique incidence; H_z and E_y for its axis-aligned twin.
    assert (len(state["electric_terms"]), len(state["magnetic_terms"])) == expected


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("residual, expected", [(0., (1, 1)), (math.cos(math.pi / 2), (1, 1)),
                                               (1e-5, (2, 2)), (1e-6, (2, 2))])
def test_mode_source_cross_product_term_counts(length, residual, expected, monkeypatch):
    from witwin.maxwell.fdtd.excitation import injection, modes

    solve = modes.solve_mode_source_profile

    def solve_with_cross_inputs(solver, compiled):
        # The public ModeSource canonicalizes its axis. Supply near-axis inputs
        # at the scalar solver boundary to exercise its physical H cross product.
        if residual < 1e-12:
            compiled = dict(compiled, direction_vector=(1., residual, residual), polarization=(residual, 1., residual))
        else:
            compiled = dict(compiled, direction_vector=(1., 0., 0.), polarization=(0., 1., residual))
        return solve(solver, compiled)

    monkeypatch.setattr(modes, "_vector_mode_supported", lambda *args, **kwargs: False)
    monkeypatch.setattr(injection, "solve_mode_source_profile", solve_with_cross_inputs)
    source = mw.ModeSource(
        position=(0., 0., 0.), size=(0., 1.2 * length, .8 * length), polarization="Ey",
        source_time=mw.CW(frequency=C_0 / (1.55 * length)), name="guided",
    )
    scene = _scene(length, source)
    scene.add_structure(mw.Structure(
        mw.Box(position=(0., 0., 0.), size=(2 * length, .45 * length, .25 * length)),
        mw.Material(eps_r=12.11), name="guide",
    ))
    solver = mw.Simulation.fdtd(scene, frequency=source.source_time.frequency).prepare().solver
    assert str(solver._compiled_sources[0]["mode_solver_kind"]).startswith("scalar")
    # The original 1e-12 floors keep H_z -> E_y and E_y -> H_z, plus
    # H_y -> E_z and E_z -> H_y only for deliberate weak polarization.
    assert (len(solver._electric_source_terms), len(solver._magnetic_source_terms)) == expected
    assert all(term["source_index"] == 0 for term in (*solver._electric_source_terms, *solver._magnetic_source_terms))
    # Rebuild the same prepared source with the original absolute floors.
    electric = list(solver._electric_source_terms)
    magnetic = list(solver._magnetic_source_terms)
    solver._electric_source_terms.clear()
    solver._magnetic_source_terms.clear()
    monkeypatch.setattr(modes, "_component_roundoff_floor", lambda components, *, dtype: 1e-12)
    monkeypatch.setattr(injection, "_component_roundoff_floor", lambda components, *, dtype: 1e-12)
    injection._prepare_mode_surface_source(solver, solver._compiled_sources[0], source_index=0)
    _assert_term_patches_equal(electric, solver._electric_source_terms)
    _assert_term_patches_equal(magnetic, solver._magnetic_source_terms)


@pytest.mark.parametrize("length", [1., 1e-6])
def test_custom_current_keeps_weak_component_and_baseline_patches(length):
    from witwin.maxwell.fdtd.excitation.injection import _yee_box_overlap

    coords = [-.4 * length, 0., .4 * length]
    components = {name: torch.full((3, 3, 3), value, dtype=torch.float64).numpy()
                  for name, value in {"Jx": 1., "Jy": 1e-6, "Jz": 0.}.items()}
    source = mw.CustomCurrentSource(
        mw.CurrentDataset((coords,) * 3, components),
        source_time=mw.CW(frequency=C_0 / (1.55 * length)), name="currents",
    )
    solver = mw.Simulation.fdtd(_scene(length, source), frequency=source.source_time.frequency).prepare().solver
    # The original 1e-30 floor retains Jx and Jy, with no magnetic current.
    assert (len(solver._source_terms), len(solver._magnetic_source_terms)) == (2, 0)
    assert [term["field_name"] for term in solver._source_terms] == ["Ex", "Ey"]
    for term, value in zip(solver._source_terms, (1., 1e-6)):
        field = term["field_name"]
        start, stop, weights = _yee_box_overlap(solver, field, (coords[0],) * 3, (coords[-1],) * 3, current_dataset=True)
        assert term["offsets"] == start
        slices = tuple(slice(lo, hi) for lo, hi in zip(start, stop))
        denominator = getattr(solver, f"eps_{field}")[slices]
        sampled = solver.Ex.new_tensor(value) * weights.to(solver.Ex.dtype)
        baseline = (-solver.dt / denominator) * sampled
        torch.testing.assert_close(term["patch"], baseline, rtol=1e-6, atol=0)
