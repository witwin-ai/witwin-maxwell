"""Actual lossy-transmission trace and reciprocity acceptance on CUDA.

These are convergence and convention checks on coarse meshes; their tolerances
do not certify the G0 independent-reference accuracy target.
"""

import json
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.fields import FieldQueryConfig, far_field
from witwin.maxwell.mom.linalg import solve_reference
from witwin.maxwell.mom.operator import IntegrationConfig, build_rhs


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA MoM transmission required")


def _solve(level, *, geometry=None, source=None):
    scene = mw.Scene(
        structures=[mw.Structure(geometry or mw.Sphere(radius=.1), mw.Material(eps_r=2.25, sigma_e=.003))],
        sources=[source or mw.PlaneWave()],
    )
    config = mw.MoMConfig(subdivisions=level, integration=IntegrationConfig(order=4, singular_order=6))
    result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    assert isinstance(result, mw.Result) and result.stats()["converged"]
    return result


def _relative_jump(outside, inside):
    denominator = (torch.linalg.vector_norm(outside)+torch.linalg.vector_norm(inside))/2
    return float(torch.linalg.vector_norm(outside-inside)/denominator.clamp_min(torch.finfo(outside.real.dtype).tiny))


def _continuity_record(result, directions):
    mesh = result.surface_currents.mesh
    centroid_directions = mesh.centroids/torch.linalg.vector_norm(mesh.centroids, dim=-1, keepdim=True)
    faces = (directions@centroid_directions.T).argmax(-1)
    points, normals = mesh.centroids[faces], mesh.normals[faces]
    options = FieldQueryConfig(relative_tolerance=1e-5, max_order=128)
    eo, ho = result.query_fields(points, region_id="exterior", field_kind="total", trace_side="exterior", query_config=options)
    ei, hi = result.query_fields(points, region_id="interior", field_kind="total", trace_side="interior", query_config=options)
    exterior, interior = result.surface_currents.handle.exterior, result.surface_currents.handle.interior
    tangent = lambda field: field-normals*(field*normals).sum(-1, keepdim=True)
    normal = lambda field: (field*normals).sum(-1)
    return {"edges": result.surface_currents.handle.basis.n_edges,
            "tangential_E_relative_jump": _relative_jump(tangent(eo), tangent(ei)),
            "tangential_H_relative_jump": _relative_jump(tangent(ho), tangent(hi)),
            "normal_D_relative_jump": _relative_jump(exterior.epsilon*normal(eo), interior.epsilon*normal(ei)),
            "normal_B_relative_jump": _relative_jump(exterior.mu*normal(ho), interior.mu*normal(hi)),
            "query_face_indices": faces.detach().cpu().tolist(),
            "query_points_m": points.detach().cpu().tolist(),
            "algebra_relative_residual": result.stats()["relative_residual"]}


def test_actual_lossy_pmchwt_one_sided_transmission_continuity(tmp_path):
    results = [_solve(level) for level in (0, 1, 2)]
    coarse = results[0].surface_currents.mesh
    directions = coarse.centroids[[0, 4, 8, 12, 16]]
    directions = directions/torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    records = [_continuity_record(result, directions) for result in results]
    (tmp_path/"transmission_trace_convergence.json").write_text(json.dumps({
        "physical_problem": "radius0.1_m_eps2.25_sigma0.003_S_per_m_frequency3e8_Hz",
        "query_selection": "triangle centroid nearest fixed coarse-face radial direction",
        "normal_convention": "outward; D uses effective complex permittivity",
        "complete_G0": False, "cases": records,
    }, indent=2)+"\n")
    print("TRANSMISSION_CONVERGENCE "+json.dumps(records))
    for key in ("tangential_E_relative_jump", "tangential_H_relative_jump", "normal_D_relative_jump", "normal_B_relative_jump"):
        errors = [record[key] for record in records]
        assert errors[2] < errors[1] < errors[0], (key, errors)
        assert errors[2] < .25, (key, errors)


def test_actual_lossy_reciprocity_reverse_directions_and_polarizations():
    result = _solve(1, geometry=mw.Ellipsoid(radii=(.1, .08, .065), position=(.03, -.02, .01)))
    handle = result.surface_currents.handle
    direction = torch.tensor([.3, .7, .4], device="cuda", dtype=torch.float64)
    direction = direction/torch.linalg.vector_norm(direction)
    polarization = torch.tensor([1., 0., 0.], device="cuda", dtype=torch.float64)
    receiver = polarization-(polarization*direction).sum()*direction
    receiver = receiver/torch.linalg.vector_norm(receiver)
    forward, _ = result.far_field(direction[None])
    reverse_source = mw.PlaneWave(direction=(-direction).tolist(), polarization=receiver.tolist())
    reverse_rhs = build_rhs(handle.basis, handle.exterior, [reverse_source], order=6)
    reverse_x, reverse_log = solve_reference(handle, reverse_rhs)
    assert reverse_log["converged"] and reverse_log["cache_reused"]
    reverse_direction = torch.tensor([[0., 0., -1.]], device="cuda", dtype=torch.float64)
    reverse, _ = far_field(handle, reverse_x, reverse_direction)
    amplitude = (forward[0]*receiver).sum()
    reciprocal = (reverse[0]*polarization).sum()
    assert float(amplitude.abs()) > 1e-5
    torch.testing.assert_close(amplitude, reciprocal, rtol=1e-4, atol=1e-10)


def test_actual_operator_complex_source_superposition_and_zero_forcing():
    result = _solve(0)
    handle = result.surface_currents.handle
    first = mw.PlaneWave(source_time=mw.CW(frequency=3e8, amplitude=2., phase=math.pi/2))
    second = mw.PlaneWave(direction=(1., 0., 0.), polarization=(0., 0., 1.),
                          source_time=mw.CW(frequency=3e8, amplitude=1.3, phase=-math.pi/3))
    opposite = mw.PlaneWave(source_time=mw.CW(frequency=3e8, amplitude=-2., phase=math.pi/2))
    rhs = torch.stack([build_rhs(handle.basis, handle.exterior, sources, order=6)
                       for sources in ([first], [second], [first, second], [first, opposite], [])])
    x, log = solve_reference(handle, rhs)
    assert log["converged"] and log["rhs_count"] == 5 and log["cache_reused"]
    torch.testing.assert_close(x[2], x[0]+x[1], rtol=2e-12, atol=2e-14)
    assert torch.count_nonzero(rhs[3:]) == torch.count_nonzero(x[3:]) == 0
    directions = torch.tensor([[1., 0., 0.], [0., 0., 1.]], device="cuda", dtype=torch.float64)
    f1, _ = far_field(handle, x[0], directions)
    f2, _ = far_field(handle, x[1], directions)
    total, _ = far_field(handle, x[2], directions)
    torch.testing.assert_close(total, f1+f2, rtol=2e-12, atol=2e-14)
