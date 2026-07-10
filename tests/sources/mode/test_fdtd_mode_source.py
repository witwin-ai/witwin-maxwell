import pytest
import torch

import witwin.maxwell as mw


def _build_mode_source_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, 0.0), size=(1.28, 0.24, 0.24)).with_material(
            mw.Material(eps_r=12.0),
            name="core",
        )
    )
    scene.add_source(
        mw.ModeSource(
            position=(-0.32, 0.0, 0.0),
            size=(0.0, 0.56, 0.56),
            polarization="Ez",
            source_time=mw.CW(frequency=1.0e9, amplitude=50.0),
            name="mode0",
        )
    )
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_mode_source_initializes_eh_surface_terms_from_guided_profile():
    solver = mw.Simulation.fdtd(_build_mode_source_scene(), frequencies=[1.0e9]).prepare().solver

    assert solver._compiled_sources[0]["kind"] == "mode_source"
    assert solver._compiled_sources[0]["effective_index"] > 1.0
    assert str(solver._compiled_sources[0]["mode_solver_kind"]).startswith("vector")
    assert solver._source_terms == []
    electric_fields = {term["field_name"] for term in solver._electric_source_terms}
    magnetic_fields = {term["field_name"] for term in solver._magnetic_source_terms}
    assert "Ez" in electric_fields
    assert "Hy" in magnetic_fields
    assert electric_fields <= {"Ey", "Ez"}
    assert magnetic_fields <= {"Hy", "Hz"}

    electric_term = next(term for term in solver._electric_source_terms if term["field_name"] == "Ez")
    patch = torch.abs(electric_term["cw_cos_patch"].squeeze(0))
    peak_value = float(torch.max(patch).item())
    edge_values = torch.cat((patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]))

    assert peak_value > 0.0
    assert peak_value > float(edge_values.mean().item()) * 2.0
